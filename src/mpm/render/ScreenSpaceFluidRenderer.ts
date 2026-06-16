import type { Camera, Matrix4 } from 'three/webgpu';

import {
  MPM_FLUID_RENDER_ALPHA,
  MPM_FLUID_RENDER_MAX_DPR,
  MPM_FLUID_RENDER_RESOLUTION_SCALE,
  MPM_FLUID_RENDER_THICKNESS_SCALE,
} from '../../config';

import depthMapWgsl from './shaders/fluidDepthMap.wgsl?raw';
import thicknessMapWgsl from './shaders/fluidThicknessMap.wgsl?raw';
import attributeMapWgsl from './shaders/fluidAttributeMap.wgsl?raw';
import fullscreenWgsl from './shaders/fluidFullscreen.wgsl?raw';
import depthFilterWgsl from './shaders/fluidDepthFilter.wgsl?raw';
import thicknessFilterWgsl from './shaders/fluidThicknessFilter.wgsl?raw';
import compositeWgsl from './shaders/fluidComposite.wgsl?raw';

interface FluidRenderTextures {
  depthMap: GPUTexture;
  tmpDepthMap: GPUTexture;
  thickness: GPUTexture;
  tmpThickness: GPUTexture;
  attribute: GPUTexture;
  tmpAttribute: GPUTexture;
  depthStencil: GPUTexture;
  depthMapView: GPUTextureView;
  tmpDepthMapView: GPUTextureView;
  thicknessView: GPUTextureView;
  tmpThicknessView: GPUTextureView;
  attributeView: GPUTextureView;
  tmpAttributeView: GPUTextureView;
  depthStencilView: GPUTextureView;
}

interface FluidRenderBindGroups {
  depthMap: GPUBindGroup;
  thicknessMap: GPUBindGroup;
  attributeMap: GPUBindGroup;
  depthFilterX: GPUBindGroup;
  depthFilterY: GPUBindGroup;
  thicknessFilterX: GPUBindGroup;
  thicknessFilterY: GPUBindGroup;
  attributeFilterX: GPUBindGroup;
  attributeFilterY: GPUBindGroup;
  composite: GPUBindGroup;
}

export type FluidDisplayType = 'none' | 'velocity' | 'density';

export interface ScreenSpaceFluidRenderParams {
  camera: Camera;
  modelMatrix: Matrix4;
  activeParticleCount: number;
  radiusWorld: number;
  thicknessScale?: number;
  displayType?: FluidDisplayType;
  displayMin?: number;
  displayMax?: number;
}

const UNIFORM_FLOATS = 92;
const FILTER_UNIFORM_FLOATS = 4;
const DEPTH_FILTER_ITERATIONS = 2;
const THICKNESS_FILTER_ITERATIONS = 1;
const GPU_BUFFER_USAGE = {
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
} as const;
const GPU_TEXTURE_USAGE = {
  TEXTURE_BINDING: 0x0004,
  RENDER_ATTACHMENT: 0x0010,
} as const;

/**
 * MLS-MPM 粒子位置 buffer を raw WebGPU で読み、透明 overlay canvas に水面を描く。
 *
 * Three.js の WebGPU render pass へ直接割り込むと backend 内部状態に依存しやすいため、
 * 同じ GPUDevice で別 canvas を重ねる。depth / thickness / blur / composite は
 * すべて GPU 上で完結し、CPU readback は行わない。
 */
export class ScreenSpaceFluidRenderer {
  private readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly presentationFormat: GPUTextureFormat;
  private readonly uniformsBuffer: GPUBuffer;
  private readonly filterXUniformBuffer: GPUBuffer;
  private readonly filterYUniformBuffer: GPUBuffer;
  private readonly uniforms = new Float32Array(UNIFORM_FLOATS);

  private particleBuffer: GPUBuffer;
  private enabled = true;
  private width = 0;
  private height = 0;
  private textures: FluidRenderTextures | undefined = undefined;
  private bindGroups: FluidRenderBindGroups | undefined = undefined;

  private readonly depthMapPipeline: GPURenderPipeline;
  private readonly thicknessMapPipeline: GPURenderPipeline;
  private readonly attributeMapPipeline: GPURenderPipeline;
  private readonly depthFilterPipeline: GPURenderPipeline;
  private readonly thicknessFilterPipeline: GPURenderPipeline;
  private readonly compositePipeline: GPURenderPipeline;

  constructor(device: GPUDevice, container: HTMLElement, particleBuffer: GPUBuffer) {
    this.device = device;
    this.particleBuffer = particleBuffer;
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.canvas = document.createElement('canvas');
    this.canvas.id = 'fluid-overlay';
    this.canvas.setAttribute('aria-hidden', 'true');
    this.canvas.style.display = 'block';
    container.appendChild(this.canvas);

    const context = this.canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (context === null) {
      throw new Error('WebGPU overlay context is unavailable');
    }
    this.context = context;

    this.uniformsBuffer = device.createBuffer({
      label: 'screen-space fluid uniforms',
      size: UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    this.filterXUniformBuffer = this.createFilterUniformBuffer([1, 0, 0, 0]);
    this.filterYUniformBuffer = this.createFilterUniformBuffer([0, 1, 0, 0]);

    const depthMapModule = device.createShaderModule({
      label: 'screen-space fluid depth map shader',
      code: depthMapWgsl,
    });
    const thicknessMapModule = device.createShaderModule({
      label: 'screen-space fluid thickness map shader',
      code: thicknessMapWgsl,
    });
    const attributeMapModule = device.createShaderModule({
      label: 'screen-space fluid attribute map shader',
      code: attributeMapWgsl,
    });
    const fullscreenModule = device.createShaderModule({
      label: 'screen-space fluid fullscreen vertex shader',
      code: fullscreenWgsl,
    });
    const depthFilterModule = device.createShaderModule({
      label: 'screen-space fluid depth filter shader',
      code: depthFilterWgsl,
    });
    const thicknessFilterModule = device.createShaderModule({
      label: 'screen-space fluid thickness filter shader',
      code: thicknessFilterWgsl,
    });
    const compositeModule = device.createShaderModule({
      label: 'screen-space fluid composite shader',
      code: compositeWgsl,
    });

    this.depthMapPipeline = device.createRenderPipeline({
      label: 'screen-space fluid depth map pipeline',
      layout: 'auto',
      vertex: { module: depthMapModule, entryPoint: 'vs' },
      fragment: {
        module: depthMapModule,
        entryPoint: 'fs',
        targets: [{ format: 'r32float' }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.thicknessMapPipeline = device.createRenderPipeline({
      label: 'screen-space fluid thickness map pipeline',
      layout: 'auto',
      vertex: { module: thicknessMapModule, entryPoint: 'vs' },
      fragment: {
        module: thicknessMapModule,
        entryPoint: 'fs',
        targets: [
          {
            format: 'r16float',
            blend: {
              color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
              alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.attributeMapPipeline = device.createRenderPipeline({
      label: 'screen-space fluid attribute map pipeline',
      layout: 'auto',
      vertex: { module: attributeMapModule, entryPoint: 'vs' },
      fragment: {
        module: attributeMapModule,
        entryPoint: 'fs',
        targets: [
          {
            format: 'r16float',
            blend: {
              color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
              alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.depthFilterPipeline = device.createRenderPipeline({
      label: 'screen-space fluid depth filter pipeline',
      layout: 'auto',
      vertex: { module: fullscreenModule, entryPoint: 'vs' },
      fragment: {
        module: depthFilterModule,
        entryPoint: 'fs',
        targets: [{ format: 'r32float' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.thicknessFilterPipeline = device.createRenderPipeline({
      label: 'screen-space fluid thickness filter pipeline',
      layout: 'auto',
      vertex: { module: fullscreenModule, entryPoint: 'vs' },
      fragment: {
        module: thicknessFilterModule,
        entryPoint: 'fs',
        targets: [{ format: 'r16float' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.compositePipeline = device.createRenderPipeline({
      label: 'screen-space fluid composite pipeline',
      layout: 'auto',
      vertex: { module: fullscreenModule, entryPoint: 'vs' },
      fragment: {
        module: compositeModule,
        entryPoint: 'fs',
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.canvas.style.display = enabled ? 'block' : 'none';
  }

  setParticleBuffer(buffer: GPUBuffer): void {
    if (this.particleBuffer === buffer) return;
    this.particleBuffer = buffer;
    this.bindGroups = this.textures ? this.createBindGroups(this.textures) : undefined;
  }

  render(params: ScreenSpaceFluidRenderParams): void {
    if (!this.enabled) return;
    this.ensureSize();

    if (params.activeParticleCount <= 0) {
      this.clearOverlay();
      return;
    }

    const textures = this.textures;
    const bindGroups = this.bindGroups;
    if (!textures || !bindGroups) return;

    const contourEnabled =
      params.displayType === 'velocity' || params.displayType === 'density';
    this.writeUniforms(params);

    const encoder = this.device.createCommandEncoder({
      label: 'screen-space fluid render encoder',
    });

    const depthPass = encoder.beginRenderPass({
      label: 'screen-space fluid depth map pass',
      colorAttachments: [
        {
          view: textures.depthMapView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: textures.depthStencilView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    depthPass.setPipeline(this.depthMapPipeline);
    depthPass.setBindGroup(0, bindGroups.depthMap);
    depthPass.draw(6, Math.floor(params.activeParticleCount));
    depthPass.end();

    for (let i = 0; i < DEPTH_FILTER_ITERATIONS; i++) {
      this.runFullscreenPass(
        encoder,
        'screen-space fluid depth filter x',
        this.depthFilterPipeline,
        bindGroups.depthFilterX,
        textures.tmpDepthMapView,
        { r: 0, g: 0, b: 0, a: 0 },
      );
      this.runFullscreenPass(
        encoder,
        'screen-space fluid depth filter y',
        this.depthFilterPipeline,
        bindGroups.depthFilterY,
        textures.depthMapView,
        { r: 0, g: 0, b: 0, a: 0 },
      );
    }

    const thicknessPass = encoder.beginRenderPass({
      label: 'screen-space fluid thickness map pass',
      colorAttachments: [
        {
          view: textures.thicknessView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    thicknessPass.setPipeline(this.thicknessMapPipeline);
    thicknessPass.setBindGroup(0, bindGroups.thicknessMap);
    thicknessPass.draw(6, Math.floor(params.activeParticleCount));
    thicknessPass.end();

    if (contourEnabled) {
      const attributePass = encoder.beginRenderPass({
        label: 'screen-space fluid attribute map pass',
        colorAttachments: [
          {
            view: textures.attributeView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      attributePass.setPipeline(this.attributeMapPipeline);
      attributePass.setBindGroup(0, bindGroups.attributeMap);
      attributePass.draw(6, Math.floor(params.activeParticleCount));
      attributePass.end();
    }

    for (let i = 0; i < THICKNESS_FILTER_ITERATIONS; i++) {
      this.runFullscreenPass(
        encoder,
        'screen-space fluid thickness filter x',
        this.thicknessFilterPipeline,
        bindGroups.thicknessFilterX,
        textures.tmpThicknessView,
        { r: 0, g: 0, b: 0, a: 0 },
      );
      this.runFullscreenPass(
        encoder,
        'screen-space fluid thickness filter y',
        this.thicknessFilterPipeline,
        bindGroups.thicknessFilterY,
        textures.thicknessView,
        { r: 0, g: 0, b: 0, a: 0 },
      );
    }

    if (contourEnabled) {
      for (let i = 0; i < THICKNESS_FILTER_ITERATIONS; i++) {
        this.runFullscreenPass(
          encoder,
          'screen-space fluid attribute filter x',
          this.thicknessFilterPipeline,
          bindGroups.attributeFilterX,
          textures.tmpAttributeView,
          { r: 0, g: 0, b: 0, a: 0 },
        );
        this.runFullscreenPass(
          encoder,
          'screen-space fluid attribute filter y',
          this.thicknessFilterPipeline,
          bindGroups.attributeFilterY,
          textures.attributeView,
          { r: 0, g: 0, b: 0, a: 0 },
        );
      }
    }

    this.runFullscreenPass(
      encoder,
      'screen-space fluid composite pass',
      this.compositePipeline,
      bindGroups.composite,
      this.context.getCurrentTexture().createView(),
      { r: 0, g: 0, b: 0, a: 0 },
    );

    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.destroyTextures();
    this.uniformsBuffer.destroy();
    this.filterXUniformBuffer.destroy();
    this.filterYUniformBuffer.destroy();
    this.canvas.remove();
  }

  private createFilterUniformBuffer(values: [number, number, number, number]): GPUBuffer {
    const buffer = this.device.createBuffer({
      label: 'screen-space fluid filter uniforms',
      size: FILTER_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, new Float32Array(values));
    return buffer;
  }

  private ensureSize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, MPM_FLUID_RENDER_MAX_DPR);
    const scale = Math.max(0.1, MPM_FLUID_RENDER_RESOLUTION_SCALE);
    const width = Math.max(1, Math.floor(window.innerWidth * dpr * scale));
    const height = Math.max(1, Math.floor(window.innerHeight * dpr * scale));

    if (width === this.width && height === this.height) return;

    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: 'premultiplied',
    });

    this.destroyTextures();
    this.textures = this.createTextures(width, height);
    this.bindGroups = this.createBindGroups(this.textures);
  }

  private createTextures(width: number, height: number): FluidRenderTextures {
    const size: [number, number, number] = [width, height, 1];
    const colorUsage =
      GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING;
    const depthMap = this.device.createTexture({
      label: 'screen-space fluid depth map texture',
      size,
      format: 'r32float',
      usage: colorUsage,
    });
    const tmpDepthMap = this.device.createTexture({
      label: 'screen-space fluid temporary depth map texture',
      size,
      format: 'r32float',
      usage: colorUsage,
    });
    const thickness = this.device.createTexture({
      label: 'screen-space fluid thickness texture',
      size,
      format: 'r16float',
      usage: colorUsage,
    });
    const tmpThickness = this.device.createTexture({
      label: 'screen-space fluid temporary thickness texture',
      size,
      format: 'r16float',
      usage: colorUsage,
    });
    const attribute = this.device.createTexture({
      label: 'screen-space fluid attribute texture',
      size,
      format: 'r16float',
      usage: colorUsage,
    });
    const tmpAttribute = this.device.createTexture({
      label: 'screen-space fluid temporary attribute texture',
      size,
      format: 'r16float',
      usage: colorUsage,
    });
    const depthStencil = this.device.createTexture({
      label: 'screen-space fluid depth stencil texture',
      size,
      format: 'depth32float',
      usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT,
    });

    return {
      depthMap,
      tmpDepthMap,
      thickness,
      tmpThickness,
      attribute,
      tmpAttribute,
      depthStencil,
      depthMapView: depthMap.createView(),
      tmpDepthMapView: tmpDepthMap.createView(),
      thicknessView: thickness.createView(),
      tmpThicknessView: tmpThickness.createView(),
      attributeView: attribute.createView(),
      tmpAttributeView: tmpAttribute.createView(),
      depthStencilView: depthStencil.createView(),
    };
  }

  private createBindGroups(textures: FluidRenderTextures): FluidRenderBindGroups {
    const depthMap = this.device.createBindGroup({
      label: 'screen-space fluid depth map bind group',
      layout: this.depthMapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const thicknessMap = this.device.createBindGroup({
      label: 'screen-space fluid thickness map bind group',
      layout: this.thicknessMapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const attributeMap = this.device.createBindGroup({
      label: 'screen-space fluid attribute map bind group',
      layout: this.attributeMapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const depthFilterX = this.device.createBindGroup({
      label: 'screen-space fluid depth filter x bind group',
      layout: this.depthFilterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.depthMapView },
        { binding: 1, resource: { buffer: this.filterXUniformBuffer } },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });
    const depthFilterY = this.device.createBindGroup({
      label: 'screen-space fluid depth filter y bind group',
      layout: this.depthFilterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.tmpDepthMapView },
        { binding: 1, resource: { buffer: this.filterYUniformBuffer } },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const thicknessFilterX = this.device.createBindGroup({
      label: 'screen-space fluid thickness filter x bind group',
      layout: this.thicknessFilterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.thicknessView },
        { binding: 1, resource: { buffer: this.filterXUniformBuffer } },
      ],
    });
    const thicknessFilterY = this.device.createBindGroup({
      label: 'screen-space fluid thickness filter y bind group',
      layout: this.thicknessFilterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.tmpThicknessView },
        { binding: 1, resource: { buffer: this.filterYUniformBuffer } },
      ],
    });

    const attributeFilterX = this.device.createBindGroup({
      label: 'screen-space fluid attribute filter x bind group',
      layout: this.thicknessFilterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.attributeView },
        { binding: 1, resource: { buffer: this.filterXUniformBuffer } },
      ],
    });
    const attributeFilterY = this.device.createBindGroup({
      label: 'screen-space fluid attribute filter y bind group',
      layout: this.thicknessFilterPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.tmpAttributeView },
        { binding: 1, resource: { buffer: this.filterYUniformBuffer } },
      ],
    });

    const composite = this.device.createBindGroup({
      label: 'screen-space fluid composite bind group',
      layout: this.compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.depthMapView },
        { binding: 1, resource: textures.thicknessView },
        { binding: 2, resource: textures.attributeView },
        { binding: 3, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    return {
      depthMap,
      thicknessMap,
      attributeMap,
      depthFilterX,
      depthFilterY,
      thicknessFilterX,
      thicknessFilterY,
      attributeFilterX,
      attributeFilterY,
      composite,
    };
  }

  private writeUniforms(params: ScreenSpaceFluidRenderParams): void {
    params.camera.updateMatrixWorld();

    let offset = 0;
    this.uniforms.set(params.camera.projectionMatrix.elements, offset);
    offset += 16;
    this.uniforms.set(params.camera.projectionMatrixInverse.elements, offset);
    offset += 16;
    this.uniforms.set(params.camera.matrixWorldInverse.elements, offset);
    offset += 16;
    this.uniforms.set(params.camera.matrixWorld.elements, offset);
    offset += 16;
    this.uniforms.set(params.modelMatrix.elements, offset);
    offset += 16;
    this.uniforms.set([
      this.width,
      this.height,
      1 / Math.max(1, this.width),
      1 / Math.max(1, this.height),
    ], offset);
    offset += 4;
    this.uniforms.set([
      Math.max(0.001, params.radiusWorld),
      MPM_FLUID_RENDER_ALPHA,
      Math.max(0.0001, params.thicknessScale ?? MPM_FLUID_RENDER_THICKNESS_SCALE),
      0,
    ], offset);
    offset += 4;
    const displayMin = Number.isFinite(params.displayMin) ? params.displayMin ?? 0 : 0;
    const requestedMax = Number.isFinite(params.displayMax) ? params.displayMax ?? 1 : 1;
    const displayMax = requestedMax > displayMin ? requestedMax : displayMin + 1e-6;
    this.uniforms.set([
      fluidDisplayTypeCode(params.displayType),
      displayMin,
      displayMax,
      0,
    ], offset);

    this.device.queue.writeBuffer(this.uniformsBuffer, 0, this.uniforms);
  }

  private runFullscreenPass(
    encoder: GPUCommandEncoder,
    label: string,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    targetView: GPUTextureView,
    clearValue: GPUColor,
  ): void {
    const pass = encoder.beginRenderPass({
      label,
      colorAttachments: [
        {
          view: targetView,
          clearValue,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  private clearOverlay(): void {
    const encoder = this.device.createCommandEncoder({
      label: 'screen-space fluid clear overlay encoder',
    });
    const pass = encoder.beginRenderPass({
      label: 'screen-space fluid clear overlay pass',
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private destroyTextures(): void {
    if (!this.textures) return;
    this.textures.depthMap.destroy();
    this.textures.tmpDepthMap.destroy();
    this.textures.thickness.destroy();
    this.textures.tmpThickness.destroy();
    this.textures.attribute.destroy();
    this.textures.tmpAttribute.destroy();
    this.textures.depthStencil.destroy();
    this.textures = undefined;
    this.bindGroups = undefined;
  }
}

function fluidDisplayTypeCode(type: FluidDisplayType | undefined): number {
  if (type === 'velocity') return 1;
  if (type === 'density') return 2;
  return 0;
}
