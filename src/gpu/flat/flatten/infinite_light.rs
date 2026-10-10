use super::super::image_infinite::ImageInfiniteDistribution;
use super::super::portal::{PortalImageInfiniteLight, PreparedPortalImage};
use super::super::texture::Mipmap;
use super::{ImageInfiniteSamplingRecord, InfiniteLightBuilder, Transform};
use crate::util::error::PbrtError;
use std::sync::Arc;

impl InfiniteLightBuilder {
    pub fn append_portal(&mut self, prepared: &PreparedPortalImage) -> Result<u32, PbrtError> {
        let distribution_offset = u32::try_from(self.portal_distribution.len())
            .map_err(|_| PbrtError::error("Portal distribution offset exceeds u32."))?;
        self.portal_distribution
            .extend_from_slice(&prepared.distribution);
        let geometry_index = u32::try_from(self.portal_records.len())
            .map_err(|_| PbrtError::error("Portal image table exceeds u32."))?;
        self.portal_records.push(PortalImageInfiniteLight {
            portal: prepared.portal,
            world_to_portal: prepared.world_to_portal,
            distribution_offset,
            resolution: prepared.resolution,
        });
        Ok(geometry_index)
    }

    pub fn append_image_distribution(
        &mut self,
        distribution: &ImageInfiniteDistribution,
        light_transform: &Transform,
    ) -> Result<u32, PbrtError> {
        let distribution_offset = u32::try_from(self.image_distribution.len())
            .map_err(|_| PbrtError::error("Image infinite distribution offset exceeds u32."))?;
        let row_cdf_offset = u32::try_from(self.image_row_cdf.len())
            .map_err(|_| PbrtError::error("Image infinite row CDF offset exceeds u32."))?;
        self.image_distribution
            .len()
            .checked_add(distribution.texels.len())
            .and_then(|end| u32::try_from(end).ok())
            .ok_or_else(|| PbrtError::error("Image infinite distribution table exceeds u32."))?;
        self.image_row_cdf
            .len()
            .checked_add(distribution.row_cdf.len())
            .and_then(|end| u32::try_from(end).ok())
            .ok_or_else(|| PbrtError::error("Image infinite row CDF table exceeds u32."))?;
        self.image_distribution
            .extend_from_slice(&distribution.texels);
        self.image_row_cdf.extend_from_slice(&distribution.row_cdf);
        let geometry_index = u32::try_from(self.image_records.len())
            .map_err(|_| PbrtError::error("Image infinite sampling record table exceeds u32."))?;
        self.image_records.push(ImageInfiniteSamplingRecord {
            distribution_offset,
            row_cdf_offset,
            resolution: distribution.resolution,
            light_to_render: [
                [
                    light_transform[0],
                    light_transform[1],
                    light_transform[2],
                    0.0,
                ],
                [
                    light_transform[4],
                    light_transform[5],
                    light_transform[6],
                    0.0,
                ],
                [
                    light_transform[8],
                    light_transform[9],
                    light_transform[10],
                    0.0,
                ],
            ],
        });
        Ok(geometry_index)
    }

    pub fn append_mipmap(&mut self, mipmap: Arc<Mipmap>) -> Result<u32, PbrtError> {
        let index = u32::try_from(self.mipmaps.len())
            .map_err(|_| PbrtError::error("Infinite light image table exceeds u32."))?;
        self.mipmaps.push(mipmap);
        Ok(index)
    }
}
