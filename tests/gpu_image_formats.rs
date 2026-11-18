#![cfg(feature = "webgpu")]

use std::sync::Arc;

use pbrt_r4::gpu::flat::texture::{
    ColorSpace, ImageCompiler, ImageDecoder, ImageValueType, Mipmap, MipmapEncoding, MipmapLevel,
    MipmapLevelData,
};
use pbrt_r4::gpu::webgpu::scene::upload::{pack_image_level, ImageUploadFormat};
use pbrt_r4::util::base::inverse_gamma_correct;

fn mipmap(channels: u32, data: MipmapLevelData, encoding: MipmapEncoding) -> Arc<Mipmap> {
    Arc::new(Mipmap {
        levels: vec![MipmapLevel {
            resolution: [1, 1],
            channels,
            data,
        }],
        color_space: ColorSpace::Srgb,
        encoding,
    })
}

#[test]
fn formats_preserve_precision_and_pad_only_rgb() {
    use wgpu::TextureFormat as F;
    let cases = [
        (
            1,
            MipmapLevelData::F16(vec![0x3555]),
            MipmapEncoding::Linear,
            F::R16Float,
            2,
        ),
        (
            1,
            MipmapLevelData::F32(vec![100_000.0]),
            MipmapEncoding::Linear,
            F::R32Float,
            4,
        ),
        (
            3,
            MipmapLevelData::F16(vec![0x3555, 0xbc00, 0x7bff]),
            MipmapEncoding::Linear,
            F::Rgba16Float,
            8,
        ),
        (
            3,
            MipmapLevelData::F32(vec![-1.0, 2.0, 100_000.0]),
            MipmapEncoding::Linear,
            F::Rgba32Float,
            16,
        ),
        (
            1,
            MipmapLevelData::U8(vec![127]),
            MipmapEncoding::Linear,
            F::R8Unorm,
            1,
        ),
        (
            3,
            MipmapLevelData::U8(vec![17, 127, 255]),
            MipmapEncoding::Linear,
            F::Rgba8Unorm,
            4,
        ),
        (
            3,
            MipmapLevelData::U8(vec![17, 127, 255]),
            MipmapEncoding::SrgbEncoded,
            F::Rgba8UnormSrgb,
            4,
        ),
    ];
    for (channels, data, encoding, format, size) in cases {
        let image = mipmap(channels, data.clone(), encoding);
        let kind = if channels == 1 {
            ImageValueType::Float
        } else {
            ImageValueType::LinearRgb
        };
        let selected = ImageUploadFormat::select(&image, kind).unwrap();
        assert_eq!(selected.format, format);
        assert_eq!(selected.bytes_per_row(3).unwrap(), size * 3);
        assert!(selected.bytes_per_row(u32::MAX).is_err() || size == 1);
        let bytes = pack_image_level(&image.levels[0]);
        assert_eq!(bytes.len(), size as usize);
        match data {
            MipmapLevelData::F16(values) => {
                let packed: Vec<_> = bytes
                    .chunks_exact(2)
                    .map(|v| u16::from_ne_bytes([v[0], v[1]]))
                    .collect();
                assert_eq!(&packed[..channels as usize], &values);
                if channels == 3 {
                    assert_eq!(packed[3], half::f16::ONE.to_bits());
                }
            }
            MipmapLevelData::F32(values) => {
                let packed: Vec<_> = bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                    .collect();
                assert_eq!(&packed[..channels as usize], &values);
                if channels == 3 {
                    assert_eq!(packed[3], 1.0);
                }
            }
            MipmapLevelData::U8(values) => {
                assert_eq!(&bytes[..channels as usize], &values);
                if channels == 3 {
                    assert_eq!(bytes[3], 255);
                }
            }
        }
    }
}

#[test]
fn upload_rejects_invalid_encoding_channels_and_mixed_mips() {
    for image in [
        mipmap(
            3,
            MipmapLevelData::F32(vec![0.0; 3]),
            MipmapEncoding::SrgbEncoded,
        ),
        mipmap(
            2,
            MipmapLevelData::F32(vec![0.0; 2]),
            MipmapEncoding::Linear,
        ),
        mipmap(
            1,
            MipmapLevelData::U8(vec![64]),
            MipmapEncoding::SrgbEncoded,
        ),
    ] {
        let kind = if image.levels[0].channels == 1 {
            ImageValueType::Float
        } else {
            ImageValueType::LinearRgb
        };
        assert!(ImageUploadFormat::select(&image, kind).is_err());
    }
    let mixed = Mipmap {
        levels: vec![
            MipmapLevel {
                resolution: [2, 1],
                channels: 1,
                data: MipmapLevelData::F32(vec![0.0; 2]),
            },
            MipmapLevel {
                resolution: [1, 1],
                channels: 1,
                data: MipmapLevelData::F16(vec![0]),
            },
        ],
        color_space: ColorSpace::Srgb,
        encoding: MipmapEncoding::Linear,
    };
    assert!(ImageUploadFormat::select(&mixed, ImageValueType::Float).is_err());
}

#[test]
fn srgb_mips_average_linear_values_and_keep_color_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("color.png");
    image::RgbImage::from_raw(2, 1, vec![0, 64, 128, 255, 128, 64])
        .unwrap()
        .save(&path)
        .unwrap();
    let source = ImageDecoder::default().decode(&path, "sRGB").unwrap();
    let compiled = ImageCompiler::default()
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap();
    assert_eq!(compiled.encoding, MipmapEncoding::SrgbEncoded);
    assert_eq!(
        compiled.levels[0].data,
        MipmapLevelData::U8(vec![0, 64, 128, 255, 128, 64])
    );
    let MipmapLevelData::U8(top) = &compiled.levels[1].data else {
        panic!("U8 mip expected")
    };
    assert_eq!(top[0], 188); // linear 0.5, encoded to sRGB
    assert!(
        (inverse_gamma_correct(top[1] as f32 / 255.0)
            - (inverse_gamma_correct(64.0 / 255.0) + inverse_gamma_correct(128.0 / 255.0)) / 2.0)
            .abs()
            < 0.002
    );
    let scalar = ImageCompiler::default()
        .compile(&source, ImageValueType::Float)
        .unwrap();
    assert_eq!(scalar.encoding, MipmapEncoding::Linear);
    assert!(!matches!(scalar.levels[0].data, MipmapLevelData::U8(_)));
}

#[test]
fn source_byte_levels_survive_decode_and_projection_exactly() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ramp.png");
    let bytes: Vec<u8> = (0..=255).collect();
    image::GrayImage::from_raw(256, 1, bytes.clone())
        .unwrap()
        .save(&path)
        .unwrap();
    for encoding in ["linear", "sRGB"] {
        let source = ImageDecoder::default().decode(&path, encoding).unwrap();
        let rgb = ImageCompiler::default()
            .compile(&source, ImageValueType::LinearRgb)
            .unwrap();
        assert_eq!(
            rgb.levels[0].data,
            MipmapLevelData::U8(bytes.iter().flat_map(|&value| [value; 3]).collect())
        );
        assert_eq!(rgb.levels.len(), 9);
    }
}

#[test]
fn alpha_keeps_v4_encoding_and_scalar_gamma_images_use_float_storage() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("alpha.png");
    image::RgbaImage::from_raw(1, 1, vec![64, 128, 192, 128])
        .unwrap()
        .save(&path)
        .unwrap();
    let source = ImageDecoder::default().decode(&path, "sRGB").unwrap();
    let alpha = ImageCompiler::default()
        .compile(&source, ImageValueType::Float)
        .unwrap();
    assert_eq!(alpha.encoding, MipmapEncoding::Linear);
    let value = match &alpha.levels[0].data {
        MipmapLevelData::F16(values) => half::f16::from_bits(values[0]).to_f32(),
        MipmapLevelData::F32(values) => values[0],
        _ => panic!("sRGB alpha must be linearized"),
    };
    assert!((value - inverse_gamma_correct(128.0 / 255.0)).abs() < 0.001);
    let gray = directory.path().join("gray.png");
    image::GrayImage::from_raw(1, 1, vec![128])
        .unwrap()
        .save(&gray)
        .unwrap();
    for encoding in ["sRGB", "gamma 2.2"] {
        let source = ImageDecoder::default().decode(&gray, encoding).unwrap();
        let scalar = ImageCompiler::default()
            .compile(&source, ImageValueType::Float)
            .unwrap();
        assert_eq!(scalar.encoding, MipmapEncoding::Linear);
        assert!(!matches!(scalar.levels[0].data, MipmapLevelData::U8(_)));
    }
}

#[test]
fn exr_half_storage_is_preserved_and_mixed_channels_stay_float() {
    use pbrt_r4::gpu::flat::texture::ImageOptimizationPolicy;
    use pbrt_r4::util::imageio::read_image_exr::{
        read_raw_image_exr_with_channels_and_metadata, read_raw_image_exr_with_storage,
    };
    use pbrt_r4::util::imageio::RawImageData;
    let directory = tempfile::tempdir().unwrap();
    let half_path = directory.path().join("half.exr");
    exr::prelude::write_rgb_file(&half_path, 2, 1, |x, _| {
        (
            half::f16::from_f32(x as f32 * 0.5),
            half::f16::ONE,
            half::f16::ZERO,
        )
    })
    .unwrap();
    assert!(matches!(
        read_raw_image_exr_with_storage(&half_path).unwrap().0.data,
        RawImageData::F16(_)
    ));
    assert!(matches!(
        read_raw_image_exr_with_channels_and_metadata(&half_path)
            .unwrap()
            .0
            .data,
        RawImageData::F32(_)
    ));
    let source = ImageDecoder::default()
        .decode(&half_path, "linear")
        .unwrap();
    let result = ImageCompiler::new(ImageOptimizationPolicy {
        allow_f16: false,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
    })
    .compile(&source, ImageValueType::LinearRgb)
    .unwrap();
    assert!(result
        .levels
        .iter()
        .all(|level| matches!(level.data, MipmapLevelData::F16(_))));
    let mixed_path = directory.path().join("mixed.exr");
    exr::prelude::write_rgb_file(&mixed_path, 1, 1, |_, _| {
        (half::f16::ONE, 100_000.0f32, half::f16::ZERO)
    })
    .unwrap();
    assert!(matches!(
        read_raw_image_exr_with_storage(&mixed_path).unwrap().0.data,
        RawImageData::F32(_)
    ));
    let source = ImageDecoder::default()
        .decode(&mixed_path, "linear")
        .unwrap();
    let result = ImageCompiler::default()
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap();
    assert!(matches!(result.levels[0].data, MipmapLevelData::F32(_)));
}

#[test]
#[ignore = "requires a WebGPU adapter with filterable F32 and texture binding arrays"]
fn mixed_formats_sample_and_load_through_one_binding_array() {
    use pbrt_r4::gpu::flat::texture::{ImageFilterMode, ImageView, ImageWrapMode};
    use pbrt_r4::gpu::webgpu::{context::Context, scene::upload::upload_texture_images};
    use wgpu::util::DeviceExt;

    let adapter = Context::request_adapter(None).unwrap();
    eprintln!("image format probe: {:?}", adapter.info);
    let (device, queue) =
        pollster::block_on(adapter.adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 7,
                ..Default::default()
            },
            required_features: wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            ..Default::default()
        }))
        .unwrap();
    let mut images = Vec::new();
    for channels in [1, 3] {
        for storage in [0, 1, 2] {
            let values = vec![0.25; 2 * channels as usize];
            let top = vec![0.25; channels as usize];
            let data = |values: Vec<f32>| match storage {
                0 => MipmapLevelData::F32(values),
                1 => MipmapLevelData::F16(
                    values
                        .into_iter()
                        .map(|v| half::f16::from_f32(v).to_bits())
                        .collect(),
                ),
                _ => MipmapLevelData::U8(values.into_iter().map(|_| 64).collect()),
            };
            images.push(Arc::new(Mipmap {
                levels: vec![
                    MipmapLevel {
                        resolution: [2, 1],
                        channels,
                        data: data(values),
                    },
                    MipmapLevel {
                        resolution: [1, 1],
                        channels,
                        data: data(top),
                    },
                ],
                color_space: ColorSpace::Srgb,
                encoding: MipmapEncoding::Linear,
            }));
        }
    }
    let srgb = Arc::make_mut(images.last_mut().unwrap()).clone();
    images.push(Arc::new(Mipmap {
        encoding: MipmapEncoding::SrgbEncoded,
        ..srgb
    }));
    let views: Vec<_> = images
        .iter()
        .enumerate()
        .map(|(index, image)| ImageView {
            mipmap: index as u32,
            value_type: if image.levels[0].channels == 1 {
                ImageValueType::Float
            } else {
                ImageValueType::LinearRgb
            },
            swrap: ImageWrapMode::Clamp,
            twrap: ImageWrapMode::Clamp,
            filter: ImageFilterMode::Trilinear,
            scale: 1.0,
            invert: false,
        })
        .collect();
    let textures = upload_texture_images(
        &device,
        &queue,
        &images,
        &views,
        &(0..images.len()).collect::<Vec<_>>(),
    )
    .unwrap();
    let texture_views: Vec<_> = textures
        .iter()
        .map(|texture| texture.create_view(&Default::default()))
        .collect();
    let texture_refs: Vec<_> = texture_views.iter().collect();
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let size = (images.len() * 4 * 16) as u64;
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &vec![0; size as usize],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: std::num::NonZeroU32::new(images.len() as u32),
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&texture_refs),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let textures_source = include_str!("../src/gpu/webgpu/shaders/lib/textures.wgsl");
    let helpers = &textures_source[textures_source.find("fn wrap_texture_index(").unwrap()
        ..textures_source.find("fn sample_texture_leaf(").unwrap()];
    let source = format!("enable wgpu_binding_array;
@group(0) @binding(0) var texture_images: binding_array<texture_2d<f32>>;\n@group(0) @binding(1) var samp: sampler;\n@group(0) @binding(2) var<storage, read_write> results: array<vec4<f32>>;\n{helpers}\n@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\nlet i = id.x;\nresults[i*4u] = textureLoad(texture_images[i], vec2<i32>(0), 0);\nresults[i*4u+1u] = textureSampleLevel(texture_images[i], samp, vec2<f32>(0.5), 0.5);\nresults[i*4u+2u] = vec4<f32>(sample_image_level(i, vec2<f32>(0.0, 0.5), 0, 1u, 2u, 2u), 1.0);\nresults[i*4u+3u] = textureLoad(texture_images[i], vec2<i32>(0), 1);\n}}");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(images.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    for (index, image) in images.iter().enumerate() {
        let expected = match index {
            2 | 5 => 64.0 / 255.0,
            6 => inverse_gamma_correct(64.0 / 255.0),
            _ => 0.25,
        };
        for mode in 0..4 {
            let expected_r = if mode == 2 { expected * 0.5 } else { expected };
            assert!(
                (values[index * 4 + mode][0] - expected_r).abs() < 0.002,
                "image {index}, mode {mode}: {:?}",
                values[index * 4 + mode]
            );
            if image.levels[0].channels == 3 {
                assert!((values[index * 4 + mode][1] - expected_r).abs() < 0.002);
            }
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn seven_image_formats_match_constant_texture_leaves() {
    use pbrt_r4::util::imageio::read_image::read_image;
    use std::fmt::Write;
    use std::process::Command;

    let temporary = tempfile::tempdir().unwrap();
    let directory = std::env::var_os("PBRT_IMAGE_FORMAT_RENDER_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temporary.path().to_owned());
    std::fs::create_dir_all(&directory).unwrap();
    image::GrayImage::from_pixel(2, 2, image::Luma([128u8]))
        .save(directory.join("gray.png"))
        .unwrap();
    image::RgbImage::from_pixel(2, 2, image::Rgb([128u8; 3]))
        .save(directory.join("rgb.png"))
        .unwrap();
    exr::prelude::write_rgb_file(directory.join("half.exr"), 2, 2, |_, _| {
        (
            half::f16::from_f32(0.5),
            half::f16::from_f32(0.5),
            half::f16::from_f32(0.5),
        )
    })
    .unwrap();
    exr::prelude::write_rgb_file(directory.join("float.exr"), 2, 2, |_, _| {
        (100_000.0f32, 100_000.0f32, 100_000.0f32)
    })
    .unwrap();

    let cases = [
        ("gray.png", "linear", true, 1.0, 128.0 / 255.0),
        ("half.exr", "linear", true, 1.0, 0.5),
        ("float.exr", "linear", true, 0.000005, 0.5),
        ("rgb.png", "linear", false, 1.0, 128.0 / 255.0),
        (
            "rgb.png",
            "sRGB",
            false,
            1.0,
            inverse_gamma_correct(128.0 / 255.0),
        ),
        ("half.exr", "linear", false, 1.0, 0.5),
        ("float.exr", "linear", false, 0.000005, 0.5),
    ];
    let mut images = Vec::new();
    for textured in [true, false] {
        let mut scene = String::from(
            r#"Integrator "volpath" "integer maxdepth" [ 1 ]
Sampler "independent" "integer pixelsamples" [ 4 ]
Film "rgb" "integer xresolution" [ 112 ] "integer yresolution" [ 32 ]
LookAt 0 0 8  0 0 0  0 1 0
Camera "perspective" "float fov" [ 35 ]
WorldBegin
LightSource "point" "point3 from" [ 0 0 6 ] "rgb I" [ 50 50 50 ]
"#,
        );
        for (index, (filename, encoding, scalar, scale, value)) in cases.iter().enumerate() {
            writeln!(scene, "AttributeBegin").unwrap();
            if textured {
                let kind = if *scalar { "float" } else { "spectrum" };
                let filter = ["point", "bilinear", "trilinear"][index % 3];
                let wrap = ["repeat", "clamp", "black"][index % 3];
                writeln!(scene, "Texture \"image-{index}\" \"{kind}\" \"imagemap\" \"string filename\" [\"{}\"] \"string encoding\" [\"{encoding}\"] \"float scale\" [{scale}] \"string filter\" [\"{filter}\"] \"string wrap\" [\"{wrap}\"]", directory.join(filename).display()).unwrap();
                if *scalar {
                    writeln!(scene, "Texture \"rgb-{index}\" \"spectrum\" \"scale\" \"rgb tex\" [0.8 0.6 0.4] \"texture scale\" [\"image-{index}\"]\nMaterial \"diffuse\" \"texture reflectance\" [\"rgb-{index}\"]").unwrap();
                } else {
                    writeln!(
                        scene,
                        "Material \"diffuse\" \"texture reflectance\" [\"image-{index}\"]"
                    )
                    .unwrap();
                }
            } else if *scalar {
                // Keep the spectrum-scale graph identical and replace only the image leaf.
                writeln!(scene, "Texture \"factor-{index}\" \"float\" \"constant\" \"float value\" [{value}]\nTexture \"rgb-{index}\" \"spectrum\" \"scale\" \"rgb tex\" [0.8 0.6 0.4] \"texture scale\" [\"factor-{index}\"]\nMaterial \"diffuse\" \"texture reflectance\" [\"rgb-{index}\"]").unwrap();
            } else {
                let rgb = [*value; 3];
                writeln!(
                    scene,
                    "Material \"diffuse\" \"rgb reflectance\" [{} {} {}]",
                    rgb[0], rgb[1], rgb[2]
                )
                .unwrap();
            }
            let x = index as f32 - 3.5;
            writeln!(scene, "Shape \"trianglemesh\" \"point3 P\" [{x} -1 0 {} -1 0 {} 1 0 {x} 1 0] \"point2 uv\" [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5] \"integer indices\" [0 1 2 0 2 3]\nAttributeEnd", x + 1.0, x + 1.0).unwrap();
        }
        let name = if textured { "textured" } else { "constant" };
        let path = directory.join(format!("{name}.pbrt"));
        let output = directory.join(format!("{name}.exr"));
        std::fs::write(&path, scene).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
            .env("RUST_LOG", "pbrt_r4::gpu=debug")
            .arg("--gpu")
            .arg("--outfile")
            .arg(&output)
            .arg(&path)
            .output()
            .unwrap();
        std::fs::write(directory.join(format!("{name}.log")), &result.stderr).unwrap();
        assert!(
            result.status.success(),
            "{name} render failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [112, 32]);
        assert!(pixels.iter().all(|pixel| pixel.is_valid()));
        images.push(pixels);
    }
    let mut max_error = 0.0f32;
    let mut max_reference = 0.0f32;
    let mut absolute_sum = 0.0f64;
    let mut energy = 0.0f64;
    for (textured, constant) in images[0].iter().zip(&images[1]) {
        for (actual, expected) in textured.to_rgb().into_iter().zip(constant.to_rgb()) {
            max_error = max_error.max((actual - expected).abs());
            max_reference = max_reference.max(expected.abs());
            absolute_sum += f64::from((actual - expected).abs());
            energy += f64::from(expected.abs());
        }
    }
    assert!(energy > 0.0);
    eprintln!(
        "seven-format render: max_abs={max_error}, max_reference={max_reference}, relative_L1={}",
        absolute_sum / energy
    );
    assert!(max_error <= 0.005 * max_reference + 1e-6);
    assert!(absolute_sum / energy < 0.005);
}

#[test]
#[ignore = "requires a native GPU adapter with allocator reports"]
fn selected_formats_reduce_native_texture_allocations() {
    use pbrt_r4::gpu::webgpu::context::Context;
    let adapter = Context::request_adapter(None).unwrap();
    eprintln!("texture allocation probe: {:?}", adapter.info);
    let (device, _) = pollster::block_on(adapter.adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::FLOAT32_FILTERABLE,
        ..Default::default()
    }))
    .unwrap();
    let allocated = || {
        device
            .generate_allocator_report()
            .expect("native allocator report required")
            .total_allocated_bytes
    };
    let before = allocated();
    let formats = [
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureFormat::R16Float,
        wgpu::TextureFormat::R32Float,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba32Float,
    ];
    let create = |format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image format allocation probe"),
            size: wgpu::Extent3d {
                width: 2048,
                height: 2048,
                depth_or_array_layers: 1,
            },
            mip_level_count: 12,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let selected: Vec<_> = formats.into_iter().map(create).collect();
    let after_selected = allocated();
    let baseline: Vec<_> = (0..7)
        .map(|_| create(wgpu::TextureFormat::Rgba32Float))
        .collect();
    let after_baseline = allocated();
    let selected_bytes = after_selected - before;
    let baseline_bytes = after_baseline - after_selected;
    assert!(selected_bytes > 0 && selected_bytes < baseline_bytes);
    eprintln!("7 images, 2048x2048, 12 mip levels: selected_allocated_bytes={selected_bytes}, rgba32f_allocated_bytes={baseline_bytes}, reduction={:.3}%", 100.0 * (1.0 - selected_bytes as f64 / baseline_bytes as f64));
    drop((selected, baseline));
}
