use super::material_eval_context::MaterialEvalContext;
use crate::base::bxdf::BxDF;
use crate::base::material::Material;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct MixMaterial {
    materials: [Arc<Material>; 2],
    amount: Arc<FloatTexture>,
}

impl MixMaterial {
    pub fn new(m0: &Arc<Material>, m1: &Arc<Material>, amount: &Arc<FloatTexture>) -> MixMaterial {
        MixMaterial {
            materials: [Arc::clone(m0), Arc::clone(m1)],
            amount: Arc::clone(amount),
        }
    }

    pub fn choose_material<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
    ) -> &Arc<Material> {
        let amt = tex_eval.evaluate_float(self.amount.as_ref(), ctx.texture_context());
        if amt <= 0.0 {
            &self.materials[0]
        } else if amt >= 1.0 {
            &self.materials[1]
        } else {
            let u = hash_mix_choice(ctx, &self.materials);
            if amt < u {
                &self.materials[0]
            } else {
                &self.materials[1]
            }
        }
    }

    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> BxDF {
        self.choose_material(tex_eval, ctx)
            .as_ref()
            .get_bxdf(tex_eval, ctx, lambda)
    }
    pub fn create(
        mp: &TextureParameterDictionary,
        m0: &Arc<Material>,
        m1: &Arc<Material>,
    ) -> Result<MixMaterial, PbrtError> {
        let amount = mp.get_float_texture("amount", 0.5)?;
        Ok(MixMaterial::new(m0, m1, &amount))
    }
}

fn hash_mix_choice(ctx: &MaterialEvalContext, materials: &[Arc<Material>; 2]) -> Float {
    let mut buf = Vec::new();
    append_float_bytes(&mut buf, ctx.texture_ctx.p.x);
    append_float_bytes(&mut buf, ctx.texture_ctx.p.y);
    append_float_bytes(&mut buf, ctx.texture_ctx.p.z);
    append_float_bytes(&mut buf, ctx.wo.x);
    append_float_bytes(&mut buf, ctx.wo.y);
    append_float_bytes(&mut buf, ctx.wo.z);
    append_usize_bytes(&mut buf, Arc::as_ptr(&materials[0]) as usize);
    append_usize_bytes(&mut buf, Arc::as_ptr(&materials[1]) as usize);
    hash_float_bytes(&buf)
}

fn append_float_bytes(buf: &mut Vec<u8>, v: Float) {
    buf.extend_from_slice(&v.to_ne_bytes());
}

fn append_usize_bytes(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&v.to_ne_bytes());
}

fn hash_float_bytes(buf: &[u8]) -> Float {
    let h = murmur_hash_64a(buf, 0);
    (h as u32) as Float * ((1.0f64 / 4_294_967_296.0f64) as Float)
}

fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    let m = 0xc6a4a7935bd1e995u64;
    let r = 47u32;

    let mut h = seed ^ ((key.len() as u64).wrapping_mul(m));

    let mut i = 0usize;
    while i + 8 <= key.len() {
        let mut k = u64::from_ne_bytes([
            key[i],
            key[i + 1],
            key[i + 2],
            key[i + 3],
            key[i + 4],
            key[i + 5],
            key[i + 6],
            key[i + 7],
        ]);
        i += 8;

        k = k.wrapping_mul(m);
        k ^= k >> r;
        k = k.wrapping_mul(m);

        h ^= k;
        h = h.wrapping_mul(m);
    }

    let tail = &key[i..];
    match tail.len() {
        7 => h ^= (tail[6] as u64) << 48,
        _ => {}
    }
    match tail.len() {
        6 | 7 => h ^= (tail[5] as u64) << 40,
        _ => {}
    }
    match tail.len() {
        5..=7 => h ^= (tail[4] as u64) << 32,
        _ => {}
    }
    match tail.len() {
        4..=7 => h ^= (tail[3] as u64) << 24,
        _ => {}
    }
    match tail.len() {
        3..=7 => h ^= (tail[2] as u64) << 16,
        _ => {}
    }
    match tail.len() {
        2..=7 => h ^= (tail[1] as u64) << 8,
        _ => {}
    }
    match tail.len() {
        1..=7 => {
            h ^= tail[0] as u64;
            h = h.wrapping_mul(m);
        }
        _ => {}
    }

    h ^= h >> r;
    h = h.wrapping_mul(m);
    h ^= h >> r;
    h
}
