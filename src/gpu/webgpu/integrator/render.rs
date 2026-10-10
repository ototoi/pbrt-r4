use std::sync::{Arc, RwLock};
use std::time::Instant;

use bytemuck::bytes_of;

use crate::displays::Display;
use crate::util::error::PbrtError;
use crate::util::misc::ProgressReporter;

use super::super::abi::{
    QueueState, QUEUE_DISPATCH_SLOT_CURRENT_RAY, QUEUE_DISPATCH_SLOT_DIRECT_EVAL,
    QUEUE_DISPATCH_SLOT_ESCAPED, QUEUE_DISPATCH_SLOT_HIT_AREA, QUEUE_DISPATCH_SLOT_MATERIAL_EVAL,
    QUEUE_DISPATCH_SLOT_MEDIUM_SCATTER, QUEUE_DISPATCH_SLOT_NEXT_RAY,
    QUEUE_DISPATCH_SLOT_SCATTER_COATED, QUEUE_DISPATCH_SLOT_SCATTER_CONDUCTOR,
    QUEUE_DISPATCH_SLOT_SCATTER_DIELECTRIC, QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE,
    QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE_TRANSMISSION, QUEUE_DISPATCH_SLOT_SCATTER_MEASURED,
    QUEUE_DISPATCH_SLOT_SCATTER_THIN_DIELECTRIC, QUEUE_DISPATCH_SLOT_SHADOW, WORKGROUP_SIZE,
};
use super::super::material::MaterialKind;
use super::super::stage::ComputeStageId;
use super::dispatch::{dispatch, dispatch_count, dispatch_indirect};
use super::tiles::{compute_tiles, tile_grid_dims, DEFAULT_DISPLAY_UPDATE_INTERVAL};

impl super::WavefrontPathIntegrator {
    fn trace_shadow_rays(&self) -> Result<(), PbrtError> {
        self.queues.reset_shadow_active(&self.context.queue);
        let mut initialize_shadow_encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 initialize shadow segment encoder"),
                });
        dispatch_indirect(
            &mut initialize_shadow_encoder,
            &self
                .pipeline
                .stage(ComputeStageId::InitializeShadowSegments)
                .pipeline,
            self.bind_groups(ComputeStageId::InitializeShadowSegments),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_SHADOW,
        );
        self.context
            .queue
            .submit(Some(initialize_shadow_encoder.finish()));
        let mut active_shadow_count = None;
        loop {
            self.queues.reset_shadow_continuation(&self.context.queue);
            let mut shadow_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pbrt-r4 shadow segment encoder"),
                    });
            if let Some(count) = active_shadow_count {
                dispatch_count(
                    &mut shadow_encoder,
                    &self
                        .pipeline
                        .stage(ComputeStageId::IntersectShadow)
                        .pipeline,
                    self.bind_groups(ComputeStageId::IntersectShadow),
                    count,
                );
            } else {
                dispatch_indirect(
                    &mut shadow_encoder,
                    &self
                        .pipeline
                        .stage(ComputeStageId::IntersectShadow)
                        .pipeline,
                    self.bind_groups(ComputeStageId::IntersectShadow),
                    &self.queues.queue_dispatch_args,
                    QUEUE_DISPATCH_SLOT_SHADOW,
                );
            }
            if self.has_interface_only_instances {
                self.queues.copy_state_to_readback(&mut shadow_encoder);
            }
            self.context.queue.submit(Some(shadow_encoder.finish()));
            if !self.has_interface_only_instances {
                break;
            }
            self.context.wait()?;
            if self.queues.read_error(&self.context.device)? {
                return Err(PbrtError::error(
                    "WebGPU wavefront rendering reported an error.",
                ));
            }
            let continuation_count = self
                .queues
                .read_shadow_continuation_count(&self.context.device)?;
            if continuation_count == 0 {
                break;
            }
            let mut copy_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pbrt-r4 shadow segment queue copy"),
                    });
            self.queues
                .copy_shadow_continuations(&mut copy_encoder, continuation_count);
            self.context.queue.submit(Some(copy_encoder.finish()));
            self.queues
                .set_shadow_active_count(&self.context.queue, continuation_count);
            active_shadow_count = Some(continuation_count);
        }
        Ok(())
    }

    fn bind_groups(&self, id: ComputeStageId) -> &[wgpu::BindGroup; 2] {
        self.bind_groups
            .get(&id)
            .expect("stage bind group is registered")
    }

    pub fn add_display(&mut self, display: &Arc<RwLock<dyn Display>>) {
        self.film.add_display(display);
    }

    fn reset_depth_queues(&self, depth: u32) {
        self.queues.reset_medium_scatter(&self.context.queue);
        if self.bssrdf_probe.is_some() {
            self.context.queue.write_buffer(
                &self.bssrdf_work,
                0,
                bytes_of(&QueueState {
                    count: 0,
                    capacity: self.tile_width * self.tile_height,
                    overflow: 0,
                    padding: 0,
                }),
            );
        }
        if depth != 0 {
            let mut reset_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pbrt-r4 wavefront queue reset encoder"),
                    });
            dispatch(
                &mut reset_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ResetShadowQueue)
                    .pipeline,
                self.bind_groups(ComputeStageId::ResetShadowQueue),
                1,
                1,
            );
            dispatch(
                &mut reset_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ResetClassificationQueues)
                    .pipeline,
                self.bind_groups(ComputeStageId::ResetClassificationQueues),
                1,
                1,
            );
            self.context.queue.submit(Some(reset_encoder.finish()));
        }
    }

    fn trace_medium_segments(&self) -> Result<(), PbrtError> {
        self.queues.reset_medium_active(&self.context.queue);
        let mut initialize_medium_encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 initialize medium segment encoder"),
                });
        dispatch_indirect(
            &mut initialize_medium_encoder,
            &self
                .pipeline
                .stage(ComputeStageId::InitializeMediumSegments)
                .pipeline,
            self.bind_groups(ComputeStageId::InitializeMediumSegments),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_CURRENT_RAY,
        );
        self.context
            .queue
            .submit(Some(initialize_medium_encoder.finish()));
        let mut active_medium_count = None;
        loop {
            self.queues.reset_medium_continuation(&self.context.queue);
            let mut segment_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pbrt-r4 medium segment encoder"),
                    });
            dispatch(
                &mut segment_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::PrepareQueueDispatch)
                    .pipeline,
                self.bind_groups(ComputeStageId::PrepareQueueDispatch),
                1,
                1,
            );
            for (pipeline, groups) in [
                (
                    &self
                        .pipeline
                        .stage(ComputeStageId::IntersectPrimaryRays)
                        .pipeline,
                    self.bind_groups(ComputeStageId::IntersectPrimaryRays),
                ),
                (
                    &self.pipeline.stage(ComputeStageId::SampleMedium).pipeline,
                    self.bind_groups(ComputeStageId::SampleMedium),
                ),
            ] {
                if let Some(count) = active_medium_count {
                    dispatch_count(&mut segment_encoder, pipeline, groups, count);
                } else {
                    dispatch_indirect(
                        &mut segment_encoder,
                        pipeline,
                        groups,
                        &self.queues.queue_dispatch_args,
                        QUEUE_DISPATCH_SLOT_CURRENT_RAY,
                    );
                }
            }
            dispatch(
                &mut segment_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::PrepareQueueDispatch)
                    .pipeline,
                self.bind_groups(ComputeStageId::PrepareQueueDispatch),
                1,
                1,
            );
            if self.has_interface_only_instances {
                self.queues.copy_state_to_readback(&mut segment_encoder);
            }
            self.context.queue.submit(Some(segment_encoder.finish()));
            if !self.has_interface_only_instances {
                break;
            }
            self.context.wait()?;
            if self.queues.read_error(&self.context.device)? {
                return Err(PbrtError::error(
                    "WebGPU wavefront rendering reported an error.",
                ));
            }
            let continuation_count = self
                .queues
                .read_medium_continuation_count(&self.context.device)?;
            if continuation_count == 0 {
                break;
            }
            let mut copy_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pbrt-r4 medium segment queue copy"),
                    });
            self.queues
                .copy_medium_continuations(&mut copy_encoder, continuation_count);
            self.context.queue.submit(Some(copy_encoder.finish()));
            self.queues
                .set_medium_active_count(&self.context.queue, continuation_count);
            active_medium_count = Some(continuation_count);
        }
        Ok(())
    }

    fn evaluate_surface(&self, depth: u32) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 wavefront surface encoder"),
                });
        dispatch_indirect(
            &mut encoder,
            &self.pipeline.stage(ComputeStageId::HandleEscaped).pipeline,
            self.bind_groups(ComputeStageId::HandleEscaped),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_ESCAPED,
        );
        // The current-ray queue's count has not changed since the
        // last prepare_queue_dispatch call, so shade_surface can
        // reuse that same slot.
        dispatch_indirect(
            &mut encoder,
            &self.pipeline.stage(ComputeStageId::ShadeSurface).pipeline,
            self.bind_groups(ComputeStageId::ShadeSurface),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_CURRENT_RAY,
        );
        // shade_surface has just finished populating the hit-area
        // and material-eval queues for this depth.
        dispatch(
            &mut encoder,
            &self
                .pipeline
                .stage(ComputeStageId::PrepareQueueDispatch)
                .pipeline,
            self.bind_groups(ComputeStageId::PrepareQueueDispatch),
            1,
            1,
        );
        dispatch_indirect(
            &mut encoder,
            &self.pipeline.stage(ComputeStageId::HandleEmissive).pipeline,
            self.bind_groups(ComputeStageId::HandleEmissive),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_HIT_AREA,
        );
        dispatch_indirect(
            &mut encoder,
            &self
                .pipeline
                .stage(ComputeStageId::EvaluateTextures)
                .pipeline,
            self.bind_groups(ComputeStageId::EvaluateTextures),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_MATERIAL_EVAL,
        );
        dispatch_indirect(
            &mut encoder,
            &self
                .pipeline
                .stage(ComputeStageId::EvaluateAttributes)
                .pipeline,
            self.bind_groups(ComputeStageId::EvaluateAttributes),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_MATERIAL_EVAL,
        );
        // classify_surface_scatter routes each hit surface into the
        // scatter queue for its resolved leaf kind, and (for the
        // non-specular kinds) the shared direct-lighting queue.
        // Direct lighting and indirect bounces both require a next
        // depth, so none of this needs to run once ray.depth reaches
        // max_depth; the queues classify_surface_scatter would fill
        // are never read at that point either way.
        if depth < self.scene.render_settings.max_depth {
            dispatch_indirect(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ClassifySurfaceScatter)
                    .pipeline,
                self.bind_groups(ComputeStageId::ClassifySurfaceScatter),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_MATERIAL_EVAL,
            );
            // The direct-eval and scatter queues' counts are final
            // now that classify_surface_scatter has run.
            dispatch(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::PrepareQueueDispatch)
                    .pipeline,
                self.bind_groups(ComputeStageId::PrepareQueueDispatch),
                1,
                1,
            );
            dispatch_indirect(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::SampleDirectLight)
                    .pipeline,
                self.bind_groups(ComputeStageId::SampleDirectLight),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_DIRECT_EVAL,
            );
            dispatch_indirect(
                &mut encoder,
                &self.pipeline.stage(ComputeStageId::ScatterMedium).pipeline,
                self.bind_groups(ComputeStageId::ScatterMedium),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_MEDIUM_SCATTER,
            );
            dispatch_indirect(
                &mut encoder,
                &self.pipeline.stage(ComputeStageId::ScatterDiffuse).pipeline,
                self.bind_groups(ComputeStageId::ScatterDiffuse),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE,
            );
            dispatch_indirect(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ScatterDiffuseTransmission)
                    .pipeline,
                self.bind_groups(ComputeStageId::ScatterDiffuseTransmission),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_SCATTER_DIFFUSE_TRANSMISSION,
            );
            dispatch_indirect(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ScatterConductor)
                    .pipeline,
                self.bind_groups(ComputeStageId::ScatterConductor),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_SCATTER_CONDUCTOR,
            );
            dispatch_indirect(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ScatterDielectric)
                    .pipeline,
                self.bind_groups(ComputeStageId::ScatterDielectric),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_SCATTER_DIELECTRIC,
            );
            dispatch_indirect(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ScatterThinDielectric)
                    .pipeline,
                self.bind_groups(ComputeStageId::ScatterThinDielectric),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_SCATTER_THIN_DIELECTRIC,
            );
            dispatch_indirect(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ScatterMeasured)
                    .pipeline,
                self.bind_groups(ComputeStageId::ScatterMeasured),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_SCATTER_MEASURED,
            );
            dispatch_indirect(
                &mut encoder,
                &self.pipeline.stage(ComputeStageId::ScatterCoated).pipeline,
                self.bind_groups(ComputeStageId::ScatterCoated),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_SCATTER_COATED,
            );
            // The scatter stages have just finished appending this
            // depth's shadow rays and next-depth bounce rays.
            dispatch(
                &mut encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::PrepareQueueDispatch)
                    .pipeline,
                self.bind_groups(ComputeStageId::PrepareQueueDispatch),
                1,
                1,
            );
            self.context.queue.submit(Some(encoder.finish()));
        } else {
            self.context.queue.submit(Some(encoder.finish()));
        }
    }

    fn sample_subsurface(&self) -> Result<(), PbrtError> {
        if let Some((probe, bindings)) = &self.bssrdf_probe {
            let mut sss_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("subsurface scattering"),
                    });
            dispatch(
                &mut sss_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ResetShadowQueue)
                    .pipeline,
                self.bind_groups(ComputeStageId::ResetShadowQueue),
                1,
                1,
            );
            dispatch(
                &mut sss_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ResetClassificationQueues)
                    .pipeline,
                self.bind_groups(ComputeStageId::ResetClassificationQueues),
                1,
                1,
            );
            let capacity = self.tile_width * self.tile_height;
            probe.encode(&mut sss_encoder, bindings, capacity);
            dispatch_count(
                &mut sss_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::PrepareSubsurfaceExit)
                    .pipeline,
                self.bind_groups(ComputeStageId::PrepareSubsurfaceExit),
                capacity,
            );
            dispatch(
                &mut sss_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::PrepareQueueDispatch)
                    .pipeline,
                self.bind_groups(ComputeStageId::PrepareQueueDispatch),
                1,
                1,
            );
            dispatch_indirect(
                &mut sss_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::SampleDirectLight)
                    .pipeline,
                self.bind_groups(ComputeStageId::SampleDirectLight),
                &self.queues.queue_dispatch_args,
                QUEUE_DISPATCH_SLOT_DIRECT_EVAL,
            );
            dispatch_count(
                &mut sss_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::ScatterSubsurfaceExit)
                    .pipeline,
                self.bind_groups(ComputeStageId::ScatterSubsurfaceExit),
                capacity,
            );
            dispatch(
                &mut sss_encoder,
                &self
                    .pipeline
                    .stage(ComputeStageId::PrepareQueueDispatch)
                    .pipeline,
                self.bind_groups(ComputeStageId::PrepareQueueDispatch),
                1,
                1,
            );
            self.context.queue.submit(Some(sss_encoder.finish()));
            self.trace_shadow_rays()?;
        }
        Ok(())
    }

    fn advance_ray_queue(&self) {
        let mut next_encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 next-ray queue encoder"),
                });
        dispatch_indirect(
            &mut next_encoder,
            &self.pipeline.stage(ComputeStageId::SwapRayQueues).pipeline,
            self.bind_groups(ComputeStageId::SwapRayQueues),
            &self.queues.queue_dispatch_args,
            QUEUE_DISPATCH_SLOT_NEXT_RAY,
        );
        dispatch(
            &mut next_encoder,
            &self
                .pipeline
                .stage(ComputeStageId::ResetNextRayQueue)
                .pipeline,
            self.bind_groups(ComputeStageId::ResetNextRayQueue),
            1,
            1,
        );
        // reset_next_ray_queue has just committed the next
        // depth's current-ray count.
        dispatch(
            &mut next_encoder,
            &self
                .pipeline
                .stage(ComputeStageId::PrepareQueueDispatch)
                .pipeline,
            self.bind_groups(ComputeStageId::PrepareQueueDispatch),
            1,
            1,
        );
        self.context.queue.submit(Some(next_encoder.finish()));
    }

    fn prepare_sample(&self, workgroups_x: u32, workgroups_y: u32) {
        let mut sample_encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 diffuse command encoder"),
                });
        dispatch(
            &mut sample_encoder,
            &self.pipeline.stage(ComputeStageId::PrepareSample).pipeline,
            self.bind_groups(ComputeStageId::PrepareSample),
            workgroups_x,
            workgroups_y,
        );
        dispatch(
            &mut sample_encoder,
            &self
                .pipeline
                .stage(ComputeStageId::GeneratePrimaryRays)
                .pipeline,
            self.bind_groups(ComputeStageId::GeneratePrimaryRays),
            workgroups_x,
            workgroups_y,
        );
        // Every pixel emits a primary ray, so the current-ray queue's
        // count is final as soon as generate_camera_rays completes.
        dispatch(
            &mut sample_encoder,
            &self
                .pipeline
                .stage(ComputeStageId::PrepareQueueDispatch)
                .pipeline,
            self.bind_groups(ComputeStageId::PrepareQueueDispatch),
            1,
            1,
        );
        self.context.queue.submit(Some(sample_encoder.finish()));
    }

    fn accumulate_sample(&self, workgroups_x: u32, workgroups_y: u32) {
        let mut accumulate_encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 sample accumulation encoder"),
                });
        dispatch(
            &mut accumulate_encoder,
            &self
                .pipeline
                .stage(ComputeStageId::AccumulateSample)
                .pipeline,
            self.bind_groups(ComputeStageId::AccumulateSample),
            workgroups_x,
            workgroups_y,
        );
        self.context.queue.submit(Some(accumulate_encoder.finish()));
    }

    pub fn render(&mut self) -> Result<(), PbrtError> {
        if self.rendered {
            return Err(PbrtError::error(
                "The initial WebGPU primary-ray integrator can only render once.",
            ));
        }
        if let Err(error) = self.film.start() {
            log::warn!("WebGPU Film display start failed: {error}");
        }
        let samples_per_pixel = self.scene.render_settings.samples_per_pixel;
        let (tiles_x, tiles_y) = tile_grid_dims(
            self.scene.viewport.region_width,
            self.scene.viewport.region_height,
            self.tile_width,
            self.tile_height,
        );
        let tile_count = tiles_x
            .checked_mul(tiles_y)
            .ok_or_else(|| PbrtError::error("GPU render: tile grid dimensions overflowed u32."))?;
        let total_iterations = tile_count
            .checked_mul(samples_per_pixel)
            .ok_or_else(|| PbrtError::error("GPU render: tile count times spp overflowed u32."))?;
        let mut reporter = self
            .show_progress
            .then(|| ProgressReporter::new(total_iterations as usize, &self.scene.output.filename));
        let mut last_display_update = Instant::now();
        log::info!(
            "GPU render: starting samples={samples_per_pixel} depth={} tiles={tile_count}",
            self.scene.render_settings.max_depth,
        );
        // Tiles are processed strictly sequentially, never in parallel: the
        // goal is bounding wavefront buffer sizes, not speed. The film
        // accumulates across all tiles, so it is only cleared once, here,
        // before any tile runs.
        {
            let mut clear_encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pbrt-r4 film clear encoder"),
                    });
            self.film.clear(&mut clear_encoder);
            self.context.queue.submit(Some(clear_encoder.finish()));
        }
        let tiles = compute_tiles(
            self.scene.viewport.region_x,
            self.scene.viewport.region_y,
            self.scene.viewport.region_width,
            self.scene.viewport.region_height,
            self.tile_width,
            self.tile_height,
        );
        for (tile_index, tile) in tiles.enumerate() {
            self.scene.viewport.tile_x = tile.x;
            self.scene.viewport.tile_y = tile.y;
            self.scene.viewport.tile_width = tile.width;
            self.scene.viewport.tile_height = tile.height;
            let workgroups_x = tile.width.div_ceil(WORKGROUP_SIZE);
            let workgroups_y = tile.height.div_ceil(WORKGROUP_SIZE);
            for sample_index in 0..samples_per_pixel {
                log::info!(
                    "GPU render: tile {}/{tile_count} sample {}/{samples_per_pixel}",
                    tile_index + 1,
                    sample_index + 1,
                );
                self.scene.viewport.sample_index = sample_index;
                self.context.queue.write_buffer(
                    &self.viewport_buffer,
                    0,
                    bytes_of(&self.scene.viewport),
                );
                self.prepare_sample(workgroups_x, workgroups_y);
                for depth in 0..=self.scene.render_settings.max_depth {
                    self.reset_depth_queues(depth);
                    self.trace_medium_segments()?;
                    self.evaluate_surface(depth);
                    if depth < self.scene.render_settings.max_depth {
                        self.trace_shadow_rays()?;
                        self.sample_subsurface()?;
                        self.advance_ray_queue();
                    }
                }
                self.accumulate_sample(workgroups_x, workgroups_y);
                log::info!(
                    "GPU render: submitted sample {sample_index}; waiting for film completion"
                );
                self.film.complete_sample()?;
                log::info!("GPU render: sample {sample_index} complete");
                let completed_iterations = self.film.completed_samples();
                if !self.film.has_no_display()
                    && (last_display_update.elapsed() >= DEFAULT_DISPLAY_UPDATE_INTERVAL
                        || completed_iterations == total_iterations)
                {
                    let mut display_encoder = self.context.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("pbrt-r4 WebGPU display readback encoder"),
                        },
                    );
                    self.film.copy_to_readback(&mut display_encoder);
                    self.queues.copy_state_to_readback(&mut display_encoder);
                    self.context.queue.submit(Some(display_encoder.finish()));
                    self.context.wait()?;
                    if self.queues.read_error(&self.context.device)? {
                        return Err(PbrtError::error(
                            "WebGPU wavefront rendering reported an error.",
                        ));
                    }
                    self.film.readback(&self.context.device)?;
                    if let Err(error) = self.film.update_display() {
                        log::warn!("WebGPU Film display update failed: {error}");
                    }
                    last_display_update = Instant::now();
                }
                if let Some(reporter) = reporter.as_mut() {
                    reporter.update(1);
                }
            }
        }
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pbrt-r4 diffuse readback encoder"),
                });
        self.film.copy_to_readback(&mut encoder);
        self.queues.copy_state_to_readback(&mut encoder);
        self.context.queue.submit(Some(encoder.finish()));
        self.context.wait()?;
        if self.queues.read_error(&self.context.device)? {
            return Err(PbrtError::error(
                "WebGPU wavefront rendering reported an error.",
            ));
        }
        self.film.readback(&self.context.device)?;
        if let Some(reporter) = reporter.as_mut() {
            reporter.done();
        }
        if let Err(error) = self.film.update_display() {
            log::warn!("WebGPU Film display update failed: {error}");
        }
        if let Err(error) = self.film.end() {
            log::warn!("WebGPU Film display end failed: {error}");
        }
        self.film.write_output(&self.scene.output)?;
        self.rendered = true;
        Ok(())
    }

    pub fn replace_material_kind(&mut self, kind: MaterialKind) {
        self.scene.replace_material_kind(kind);
        self.context.queue.write_buffer(
            &self.material_table_buffer,
            0,
            bytes_of(&self.scene.material_table),
        );
    }
}
