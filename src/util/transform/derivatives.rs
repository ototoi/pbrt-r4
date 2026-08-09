use crate::util::base::*;
use crate::util::quaternion::*;
use crate::util::transform::*;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct DerivativeTerm([Float; 4]);

impl DerivativeTerm {
    pub fn zero() -> Self {
        DerivativeTerm([0.0, 0.0, 0.0, 0.0])
    }
    pub fn new(c: Float, x: Float, y: Float, z: Float) -> Self {
        DerivativeTerm([c, x, y, z])
    }
    pub fn eval(&self, p: &Point3f) -> Float {
        return self.0[0] + self.0[1] * p.x + self.0[2] * p.y + self.0[3] * p.z;
    }
}

pub fn get_derivatives(
    t: &[Vector3f; 2],
    r: &[Quaternion; 2],
    s: &[Matrix4x4; 2],
) -> [[DerivativeTerm; 3]; 5] {
    let cos_theta = Quaternion::dot(&r[0], &r[1]);
    let theta = Float::acos(Float::clamp(cos_theta, -1.0, 1.0));
    let qperp = Quaternion::normalize(&(r[1] - r[0] * cos_theta));

    let t0x = t[0].x;
    let t0y = t[0].y;
    let t0z = t[0].z;
    let t1x = t[1].x;
    let t1y = t[1].y;
    let t1z = t[1].z;
    let q0x = r[0].x;
    let q0y = r[0].y;
    let q0z = r[0].z;
    let q0w = r[0].w;

    let qdx = qperp.x; //qperpx -> qsx
    let qdy = qperp.y; //qperpx -> qsx
    let qdz = qperp.z; //qperpx -> qsx
    let qdw = qperp.w; //qperpx -> qsx

    let s000 = s[0].m[4 * 0 + 0];
    let s001 = s[0].m[4 * 0 + 1];
    let s002 = s[0].m[4 * 0 + 2];
    let s010 = s[0].m[4 * 1 + 0];
    let s011 = s[0].m[4 * 1 + 1];
    let s012 = s[0].m[4 * 1 + 2];
    let s020 = s[0].m[4 * 2 + 0];
    let s021 = s[0].m[4 * 2 + 1];
    let s022 = s[0].m[4 * 2 + 2];

    let s100 = s[1].m[4 * 0 + 0];
    let s101 = s[1].m[4 * 0 + 1];
    let s102 = s[1].m[4 * 0 + 2];
    let s110 = s[1].m[4 * 1 + 0];
    let s111 = s[1].m[4 * 1 + 1];
    let s112 = s[1].m[4 * 1 + 2];
    let s120 = s[1].m[4 * 2 + 0];
    let s121 = s[1].m[4 * 2 + 1];
    let s122 = s[1].m[4 * 2 + 2];

    let c1_0 = DerivativeTerm::new(
        -t0x + t1x,
        (-1.0 + q0y * q0y + q0z * q0z + qdy * qdy + qdz * qdz) * s000 + q0w * q0z * s010
            - qdx * qdy * s010
            + qdw * qdz * s010
            - q0w * q0y * s020
            - qdw * qdy * s020
            - qdx * qdz * s020
            + s100
            - q0y * q0y * s100
            - q0z * q0z * s100
            - qdy * qdy * s100
            - qdz * qdz * s100
            - q0w * q0z * s110
            + qdx * qdy * s110
            - qdw * qdz * s110
            + q0w * q0y * s120
            + qdw * qdy * s120
            + qdx * qdz * s120
            + q0x * (-(q0y * s010) - q0z * s020 + q0y * s110 + q0z * s120),
        (-1.0 + q0y * q0y + q0z * q0z + qdy * qdy + qdz * qdz) * s001 + q0w * q0z * s011
            - qdx * qdy * s011
            + qdw * qdz * s011
            - q0w * q0y * s021
            - qdw * qdy * s021
            - qdx * qdz * s021
            + s101
            - q0y * q0y * s101
            - q0z * q0z * s101
            - qdy * qdy * s101
            - qdz * qdz * s101
            - q0w * q0z * s111
            + qdx * qdy * s111
            - qdw * qdz * s111
            + q0w * q0y * s121
            + qdw * qdy * s121
            + qdx * qdz * s121
            + q0x * (-(q0y * s011) - q0z * s021 + q0y * s111 + q0z * s121),
        (-1.0 + q0y * q0y + q0z * q0z + qdy * qdy + qdz * qdz) * s002 + q0w * q0z * s012
            - qdx * qdy * s012
            + qdw * qdz * s012
            - q0w * q0y * s022
            - qdw * qdy * s022
            - qdx * qdz * s022
            + s102
            - q0y * q0y * s102
            - q0z * q0z * s102
            - qdy * qdy * s102
            - qdz * qdz * s102
            - q0w * q0z * s112
            + qdx * qdy * s112
            - qdw * qdz * s112
            + q0w * q0y * s122
            + qdw * qdy * s122
            + qdx * qdz * s122
            + q0x * (-(q0y * s012) - q0z * s022 + q0y * s112 + q0z * s122),
    );

    let c2_0 = DerivativeTerm::new(
        0.0,
        -(qdy * qdy * s000) - qdz * qdz * s000 + qdx * qdy * s010 - qdw * qdz * s010
            + qdw * qdy * s020
            + qdx * qdz * s020
            + q0y * q0y * (s000 - s100)
            + q0z * q0z * (s000 - s100)
            + qdy * qdy * s100
            + qdz * qdz * s100
            - qdx * qdy * s110
            + qdw * qdz * s110
            - qdw * qdy * s120
            - qdx * qdz * s120
            + 2.0 * q0x * qdy * s010 * theta
            - 2.0 * q0w * qdz * s010 * theta
            + 2.0 * q0w * qdy * s020 * theta
            + 2.0 * q0x * qdz * s020 * theta
            + q0y
                * (q0x * (-s010 + s110)
                    + q0w * (-s020 + s120)
                    + 2.0 * (-2.0 * qdy * s000 + qdx * s010 + qdw * s020) * theta)
            + q0z
                * (q0w * (s010 - s110) + q0x * (-s020 + s120)
                    - 2.0 * (2.0 * qdz * s000 + qdw * s010 - qdx * s020) * theta),
        -(qdy * qdy * s001) - qdz * qdz * s001 + qdx * qdy * s011 - qdw * qdz * s011
            + qdw * qdy * s021
            + qdx * qdz * s021
            + q0y * q0y * (s001 - s101)
            + q0z * q0z * (s001 - s101)
            + qdy * qdy * s101
            + qdz * qdz * s101
            - qdx * qdy * s111
            + qdw * qdz * s111
            - qdw * qdy * s121
            - qdx * qdz * s121
            + 2.0 * q0x * qdy * s011 * theta
            - 2.0 * q0w * qdz * s011 * theta
            + 2.0 * q0w * qdy * s021 * theta
            + 2.0 * q0x * qdz * s021 * theta
            + q0y
                * (q0x * (-s011 + s111)
                    + q0w * (-s021 + s121)
                    + 2.0 * (-2.0 * qdy * s001 + qdx * s011 + qdw * s021) * theta)
            + q0z
                * (q0w * (s011 - s111) + q0x * (-s021 + s121)
                    - 2.0 * (2.0 * qdz * s001 + qdw * s011 - qdx * s021) * theta),
        -(qdy * qdy * s002) - qdz * qdz * s002 + qdx * qdy * s012 - qdw * qdz * s012
            + qdw * qdy * s022
            + qdx * qdz * s022
            + q0y * q0y * (s002 - s102)
            + q0z * q0z * (s002 - s102)
            + qdy * qdy * s102
            + qdz * qdz * s102
            - qdx * qdy * s112
            + qdw * qdz * s112
            - qdw * qdy * s122
            - qdx * qdz * s122
            + 2.0 * q0x * qdy * s012 * theta
            - 2.0 * q0w * qdz * s012 * theta
            + 2.0 * q0w * qdy * s022 * theta
            + 2.0 * q0x * qdz * s022 * theta
            + q0y
                * (q0x * (-s012 + s112)
                    + q0w * (-s022 + s122)
                    + 2.0 * (-2.0 * qdy * s002 + qdx * s012 + qdw * s022) * theta)
            + q0z
                * (q0w * (s012 - s112) + q0x * (-s022 + s122)
                    - 2.0 * (2.0 * qdz * s002 + qdw * s012 - qdx * s022) * theta),
    );

    let c3_0 = DerivativeTerm::new(
        0.0,
        -2.0 * (q0x * qdy * s010 - q0w * qdz * s010 + q0w * qdy * s020 + q0x * qdz * s020
            - q0x * qdy * s110
            + q0w * qdz * s110
            - q0w * qdy * s120
            - q0x * qdz * s120
            + q0y
                * (-2.0 * qdy * s000 + qdx * s010 + qdw * s020 + 2.0 * qdy * s100
                    - qdx * s110
                    - qdw * s120)
            + q0z
                * (-2.0 * qdz * s000 - qdw * s010 + qdx * s020 + 2.0 * qdz * s100 + qdw * s110
                    - qdx * s120))
            * theta,
        -2.0 * (q0x * qdy * s011 - q0w * qdz * s011 + q0w * qdy * s021 + q0x * qdz * s021
            - q0x * qdy * s111
            + q0w * qdz * s111
            - q0w * qdy * s121
            - q0x * qdz * s121
            + q0y
                * (-2.0 * qdy * s001 + qdx * s011 + qdw * s021 + 2.0 * qdy * s101
                    - qdx * s111
                    - qdw * s121)
            + q0z
                * (-2.0 * qdz * s001 - qdw * s011 + qdx * s021 + 2.0 * qdz * s101 + qdw * s111
                    - qdx * s121))
            * theta,
        -2.0 * (q0x * qdy * s012 - q0w * qdz * s012 + q0w * qdy * s022 + q0x * qdz * s022
            - q0x * qdy * s112
            + q0w * qdz * s112
            - q0w * qdy * s122
            - q0x * qdz * s122
            + q0y
                * (-2.0 * qdy * s002 + qdx * s012 + qdw * s022 + 2.0 * qdy * s102
                    - qdx * s112
                    - qdw * s122)
            + q0z
                * (-2.0 * qdz * s002 - qdw * s012 + qdx * s022 + 2.0 * qdz * s102 + qdw * s112
                    - qdx * s122))
            * theta,
    );

    let c4_0 = DerivativeTerm::new(
        0.0,
        -(q0x * qdy * s010) + q0w * qdz * s010 - q0w * qdy * s020 - q0x * qdz * s020
            + q0x * qdy * s110
            - q0w * qdz * s110
            + q0w * qdy * s120
            + q0x * qdz * s120
            + 2.0 * q0y * q0y * s000 * theta
            + 2.0 * q0z * q0z * s000 * theta
            - 2.0 * qdy * qdy * s000 * theta
            - 2.0 * qdz * qdz * s000 * theta
            + 2.0 * qdx * qdy * s010 * theta
            - 2.0 * qdw * qdz * s010 * theta
            + 2.0 * qdw * qdy * s020 * theta
            + 2.0 * qdx * qdz * s020 * theta
            + q0y
                * (-(qdx * s010) - qdw * s020
                    + 2.0 * qdy * (s000 - s100)
                    + qdx * s110
                    + qdw * s120
                    - 2.0 * q0x * s010 * theta
                    - 2.0 * q0w * s020 * theta)
            + q0z
                * (2.0 * qdz * s000 + qdw * s010 - qdx * s020 - 2.0 * qdz * s100 - qdw * s110
                    + qdx * s120
                    + 2.0 * q0w * s010 * theta
                    - 2.0 * q0x * s020 * theta),
        -(q0x * qdy * s011) + q0w * qdz * s011 - q0w * qdy * s021 - q0x * qdz * s021
            + q0x * qdy * s111
            - q0w * qdz * s111
            + q0w * qdy * s121
            + q0x * qdz * s121
            + 2.0 * q0y * q0y * s001 * theta
            + 2.0 * q0z * q0z * s001 * theta
            - 2.0 * qdy * qdy * s001 * theta
            - 2.0 * qdz * qdz * s001 * theta
            + 2.0 * qdx * qdy * s011 * theta
            - 2.0 * qdw * qdz * s011 * theta
            + 2.0 * qdw * qdy * s021 * theta
            + 2.0 * qdx * qdz * s021 * theta
            + q0y
                * (-(qdx * s011) - qdw * s021
                    + 2.0 * qdy * (s001 - s101)
                    + qdx * s111
                    + qdw * s121
                    - 2.0 * q0x * s011 * theta
                    - 2.0 * q0w * s021 * theta)
            + q0z
                * (2.0 * qdz * s001 + qdw * s011 - qdx * s021 - 2.0 * qdz * s101 - qdw * s111
                    + qdx * s121
                    + 2.0 * q0w * s011 * theta
                    - 2.0 * q0x * s021 * theta),
        -(q0x * qdy * s012) + q0w * qdz * s012 - q0w * qdy * s022 - q0x * qdz * s022
            + q0x * qdy * s112
            - q0w * qdz * s112
            + q0w * qdy * s122
            + q0x * qdz * s122
            + 2.0 * q0y * q0y * s002 * theta
            + 2.0 * q0z * q0z * s002 * theta
            - 2.0 * qdy * qdy * s002 * theta
            - 2.0 * qdz * qdz * s002 * theta
            + 2.0 * qdx * qdy * s012 * theta
            - 2.0 * qdw * qdz * s012 * theta
            + 2.0 * qdw * qdy * s022 * theta
            + 2.0 * qdx * qdz * s022 * theta
            + q0y
                * (-(qdx * s012) - qdw * s022
                    + 2.0 * qdy * (s002 - s102)
                    + qdx * s112
                    + qdw * s122
                    - 2.0 * q0x * s012 * theta
                    - 2.0 * q0w * s022 * theta)
            + q0z
                * (2.0 * qdz * s002 + qdw * s012 - qdx * s022 - 2.0 * qdz * s102 - qdw * s112
                    + qdx * s122
                    + 2.0 * q0w * s012 * theta
                    - 2.0 * q0x * s022 * theta),
    );

    let c5_0 = DerivativeTerm::new(
        0.0,
        2.0 * (qdy * qdy * s000 + qdz * qdz * s000 - qdx * qdy * s010 + qdw * qdz * s010
            - qdw * qdy * s020
            - qdx * qdz * s020
            - qdy * qdy * s100
            - qdz * qdz * s100
            + q0y * q0y * (-s000 + s100)
            + q0z * q0z * (-s000 + s100)
            + qdx * qdy * s110
            - qdw * qdz * s110
            + q0y * (q0x * (s010 - s110) + q0w * (s020 - s120))
            + qdw * qdy * s120
            + qdx * qdz * s120
            + q0z * (-(q0w * s010) + q0x * s020 + q0w * s110 - q0x * s120))
            * theta,
        2.0 * (qdy * qdy * s001 + qdz * qdz * s001 - qdx * qdy * s011 + qdw * qdz * s011
            - qdw * qdy * s021
            - qdx * qdz * s021
            - qdy * qdy * s101
            - qdz * qdz * s101
            + q0y * q0y * (-s001 + s101)
            + q0z * q0z * (-s001 + s101)
            + qdx * qdy * s111
            - qdw * qdz * s111
            + q0y * (q0x * (s011 - s111) + q0w * (s021 - s121))
            + qdw * qdy * s121
            + qdx * qdz * s121
            + q0z * (-(q0w * s011) + q0x * s021 + q0w * s111 - q0x * s121))
            * theta,
        2.0 * (qdy * qdy * s002 + qdz * qdz * s002 - qdx * qdy * s012 + qdw * qdz * s012
            - qdw * qdy * s022
            - qdx * qdz * s022
            - qdy * qdy * s102
            - qdz * qdz * s102
            + q0y * q0y * (-s002 + s102)
            + q0z * q0z * (-s002 + s102)
            + qdx * qdy * s112
            - qdw * qdz * s112
            + q0y * (q0x * (s012 - s112) + q0w * (s022 - s122))
            + qdw * qdy * s122
            + qdx * qdz * s122
            + q0z * (-(q0w * s012) + q0x * s022 + q0w * s112 - q0x * s122))
            * theta,
    );

    let c1_1 = DerivativeTerm::new(
        -t0y + t1y,
        -(qdx * qdy * s000) - qdw * qdz * s000 - s010
            + q0z * q0z * s010
            + qdx * qdx * s010
            + qdz * qdz * s010
            - q0y * q0z * s020
            + qdw * qdx * s020
            - qdy * qdz * s020
            + qdx * qdy * s100
            + qdw * qdz * s100
            + q0w * q0z * (-s000 + s100)
            + q0x * q0x * (s010 - s110)
            + s110
            - q0z * q0z * s110
            - qdx * qdx * s110
            - qdz * qdz * s110
            + q0x * (q0y * (-s000 + s100) + q0w * (s020 - s120))
            + q0y * q0z * s120
            - qdw * qdx * s120
            + qdy * qdz * s120,
        -(qdx * qdy * s001) - qdw * qdz * s001 - s011
            + q0z * q0z * s011
            + qdx * qdx * s011
            + qdz * qdz * s011
            - q0y * q0z * s021
            + qdw * qdx * s021
            - qdy * qdz * s021
            + qdx * qdy * s101
            + qdw * qdz * s101
            + q0w * q0z * (-s001 + s101)
            + q0x * q0x * (s011 - s111)
            + s111
            - q0z * q0z * s111
            - qdx * qdx * s111
            - qdz * qdz * s111
            + q0x * (q0y * (-s001 + s101) + q0w * (s021 - s121))
            + q0y * q0z * s121
            - qdw * qdx * s121
            + qdy * qdz * s121,
        -(qdx * qdy * s002) - qdw * qdz * s002 - s012
            + q0z * q0z * s012
            + qdx * qdx * s012
            + qdz * qdz * s012
            - q0y * q0z * s022
            + qdw * qdx * s022
            - qdy * qdz * s022
            + qdx * qdy * s102
            + qdw * qdz * s102
            + q0w * q0z * (-s002 + s102)
            + q0x * q0x * (s012 - s112)
            + s112
            - q0z * q0z * s112
            - qdx * qdx * s112
            - qdz * qdz * s112
            + q0x * (q0y * (-s002 + s102) + q0w * (s022 - s122))
            + q0y * q0z * s122
            - qdw * qdx * s122
            + qdy * qdz * s122,
    );

    let c2_1 = DerivativeTerm::new(
        0.0,
        qdx * qdy * s000 + qdw * qdz * s000 + q0z * q0z * s010
            - qdx * qdx * s010
            - qdz * qdz * s010
            - q0y * q0z * s020
            - qdw * qdx * s020
            + qdy * qdz * s020
            - qdx * qdy * s100
            - qdw * qdz * s100
            + q0x * q0x * (s010 - s110)
            - q0z * q0z * s110
            + qdx * qdx * s110
            + qdz * qdz * s110
            + q0y * q0z * s120
            + qdw * qdx * s120
            - qdy * qdz * s120
            + 2.0 * q0z * qdw * s000 * theta
            + 2.0 * q0y * qdx * s000 * theta
            - 4.0 * q0z * qdz * s010 * theta
            + 2.0 * q0z * qdy * s020 * theta
            + 2.0 * q0y * qdz * s020 * theta
            + q0x
                * (q0w * s020 + q0y * (-s000 + s100) - q0w * s120 + 2.0 * qdy * s000 * theta
                    - 4.0 * qdx * s010 * theta
                    - 2.0 * qdw * s020 * theta)
            + q0w
                * (-(q0z * s000) + q0z * s100 + 2.0 * qdz * s000 * theta
                    - 2.0 * qdx * s020 * theta),
        qdx * qdy * s001 + qdw * qdz * s001 + q0z * q0z * s011
            - qdx * qdx * s011
            - qdz * qdz * s011
            - q0y * q0z * s021
            - qdw * qdx * s021
            + qdy * qdz * s021
            - qdx * qdy * s101
            - qdw * qdz * s101
            + q0x * q0x * (s011 - s111)
            - q0z * q0z * s111
            + qdx * qdx * s111
            + qdz * qdz * s111
            + q0y * q0z * s121
            + qdw * qdx * s121
            - qdy * qdz * s121
            + 2.0 * q0z * qdw * s001 * theta
            + 2.0 * q0y * qdx * s001 * theta
            - 4.0 * q0z * qdz * s011 * theta
            + 2.0 * q0z * qdy * s021 * theta
            + 2.0 * q0y * qdz * s021 * theta
            + q0x
                * (q0w * s021 + q0y * (-s001 + s101) - q0w * s121 + 2.0 * qdy * s001 * theta
                    - 4.0 * qdx * s011 * theta
                    - 2.0 * qdw * s021 * theta)
            + q0w
                * (-(q0z * s001) + q0z * s101 + 2.0 * qdz * s001 * theta
                    - 2.0 * qdx * s021 * theta),
        qdx * qdy * s002 + qdw * qdz * s002 + q0z * q0z * s012
            - qdx * qdx * s012
            - qdz * qdz * s012
            - q0y * q0z * s022
            - qdw * qdx * s022
            + qdy * qdz * s022
            - qdx * qdy * s102
            - qdw * qdz * s102
            + q0x * q0x * (s012 - s112)
            - q0z * q0z * s112
            + qdx * qdx * s112
            + qdz * qdz * s112
            + q0y * q0z * s122
            + qdw * qdx * s122
            - qdy * qdz * s122
            + 2.0 * q0z * qdw * s002 * theta
            + 2.0 * q0y * qdx * s002 * theta
            - 4.0 * q0z * qdz * s012 * theta
            + 2.0 * q0z * qdy * s022 * theta
            + 2.0 * q0y * qdz * s022 * theta
            + q0x
                * (q0w * s022 + q0y * (-s002 + s102) - q0w * s122 + 2.0 * qdy * s002 * theta
                    - 4.0 * qdx * s012 * theta
                    - 2.0 * qdw * s022 * theta)
            + q0w
                * (-(q0z * s002) + q0z * s102 + 2.0 * qdz * s002 * theta
                    - 2.0 * qdx * s022 * theta),
    );

    let c3_1 = DerivativeTerm::new(
        0.0,
        2.0 * (-(q0x * qdy * s000) - q0w * qdz * s000
            + 2.0 * q0x * qdx * s010
            + q0x * qdw * s020
            + q0w * qdx * s020
            + q0x * qdy * s100
            + q0w * qdz * s100
            - 2.0 * q0x * qdx * s110
            - q0x * qdw * s120
            - q0w * qdx * s120
            + q0z
                * (2.0 * qdz * s010 - qdy * s020 + qdw * (-s000 + s100) - 2.0 * qdz * s110
                    + qdy * s120)
            + q0y * (-(qdx * s000) - qdz * s020 + qdx * s100 + qdz * s120))
            * theta,
        2.0 * (-(q0x * qdy * s001) - q0w * qdz * s001
            + 2.0 * q0x * qdx * s011
            + q0x * qdw * s021
            + q0w * qdx * s021
            + q0x * qdy * s101
            + q0w * qdz * s101
            - 2.0 * q0x * qdx * s111
            - q0x * qdw * s121
            - q0w * qdx * s121
            + q0z
                * (2.0 * qdz * s011 - qdy * s021 + qdw * (-s001 + s101) - 2.0 * qdz * s111
                    + qdy * s121)
            + q0y * (-(qdx * s001) - qdz * s021 + qdx * s101 + qdz * s121))
            * theta,
        2.0 * (-(q0x * qdy * s002) - q0w * qdz * s002
            + 2.0 * q0x * qdx * s012
            + q0x * qdw * s022
            + q0w * qdx * s022
            + q0x * qdy * s102
            + q0w * qdz * s102
            - 2.0 * q0x * qdx * s112
            - q0x * qdw * s122
            - q0w * qdx * s122
            + q0z
                * (2.0 * qdz * s012 - qdy * s022 + qdw * (-s002 + s102) - 2.0 * qdz * s112
                    + qdy * s122)
            + q0y * (-(qdx * s002) - qdz * s022 + qdx * s102 + qdz * s122))
            * theta,
    );

    let c4_1 = DerivativeTerm::new(
        0.0,
        -(q0x * qdy * s000) - q0w * qdz * s000
            + 2.0 * q0x * qdx * s010
            + q0x * qdw * s020
            + q0w * qdx * s020
            + q0x * qdy * s100
            + q0w * qdz * s100
            - 2.0 * q0x * qdx * s110
            - q0x * qdw * s120
            - q0w * qdx * s120
            + 2.0 * qdx * qdy * s000 * theta
            + 2.0 * qdw * qdz * s000 * theta
            + 2.0 * q0x * q0x * s010 * theta
            + 2.0 * q0z * q0z * s010 * theta
            - 2.0 * qdx * qdx * s010 * theta
            - 2.0 * qdz * qdz * s010 * theta
            + 2.0 * q0w * q0x * s020 * theta
            - 2.0 * qdw * qdx * s020 * theta
            + 2.0 * qdy * qdz * s020 * theta
            + q0y
                * (-(qdx * s000) - qdz * s020 + qdx * s100 + qdz * s120 - 2.0 * q0x * s000 * theta)
            + q0z
                * (2.0 * qdz * s010 - qdy * s020 + qdw * (-s000 + s100) - 2.0 * qdz * s110
                    + qdy * s120
                    - 2.0 * q0w * s000 * theta
                    - 2.0 * q0y * s020 * theta),
        -(q0x * qdy * s001) - q0w * qdz * s001
            + 2.0 * q0x * qdx * s011
            + q0x * qdw * s021
            + q0w * qdx * s021
            + q0x * qdy * s101
            + q0w * qdz * s101
            - 2.0 * q0x * qdx * s111
            - q0x * qdw * s121
            - q0w * qdx * s121
            + 2.0 * qdx * qdy * s001 * theta
            + 2.0 * qdw * qdz * s001 * theta
            + 2.0 * q0x * q0x * s011 * theta
            + 2.0 * q0z * q0z * s011 * theta
            - 2.0 * qdx * qdx * s011 * theta
            - 2.0 * qdz * qdz * s011 * theta
            + 2.0 * q0w * q0x * s021 * theta
            - 2.0 * qdw * qdx * s021 * theta
            + 2.0 * qdy * qdz * s021 * theta
            + q0y
                * (-(qdx * s001) - qdz * s021 + qdx * s101 + qdz * s121 - 2.0 * q0x * s001 * theta)
            + q0z
                * (2.0 * qdz * s011 - qdy * s021 + qdw * (-s001 + s101) - 2.0 * qdz * s111
                    + qdy * s121
                    - 2.0 * q0w * s001 * theta
                    - 2.0 * q0y * s021 * theta),
        -(q0x * qdy * s002) - q0w * qdz * s002
            + 2.0 * q0x * qdx * s012
            + q0x * qdw * s022
            + q0w * qdx * s022
            + q0x * qdy * s102
            + q0w * qdz * s102
            - 2.0 * q0x * qdx * s112
            - q0x * qdw * s122
            - q0w * qdx * s122
            + 2.0 * qdx * qdy * s002 * theta
            + 2.0 * qdw * qdz * s002 * theta
            + 2.0 * q0x * q0x * s012 * theta
            + 2.0 * q0z * q0z * s012 * theta
            - 2.0 * qdx * qdx * s012 * theta
            - 2.0 * qdz * qdz * s012 * theta
            + 2.0 * q0w * q0x * s022 * theta
            - 2.0 * qdw * qdx * s022 * theta
            + 2.0 * qdy * qdz * s022 * theta
            + q0y
                * (-(qdx * s002) - qdz * s022 + qdx * s102 + qdz * s122 - 2.0 * q0x * s002 * theta)
            + q0z
                * (2.0 * qdz * s012 - qdy * s022 + qdw * (-s002 + s102) - 2.0 * qdz * s112
                    + qdy * s122
                    - 2.0 * q0w * s002 * theta
                    - 2.0 * q0y * s022 * theta),
    );

    let c5_1 = DerivativeTerm::new(
        0.0,
        -2.0 * (qdx * qdy * s000 + qdw * qdz * s000 + q0z * q0z * s010
            - qdx * qdx * s010
            - qdz * qdz * s010
            - q0y * q0z * s020
            - qdw * qdx * s020
            + qdy * qdz * s020
            - qdx * qdy * s100
            - qdw * qdz * s100
            + q0w * q0z * (-s000 + s100)
            + q0x * q0x * (s010 - s110)
            - q0z * q0z * s110
            + qdx * qdx * s110
            + qdz * qdz * s110
            + q0x * (q0y * (-s000 + s100) + q0w * (s020 - s120))
            + q0y * q0z * s120
            + qdw * qdx * s120
            - qdy * qdz * s120)
            * theta,
        -2.0 * (qdx * qdy * s001 + qdw * qdz * s001 + q0z * q0z * s011
            - qdx * qdx * s011
            - qdz * qdz * s011
            - q0y * q0z * s021
            - qdw * qdx * s021
            + qdy * qdz * s021
            - qdx * qdy * s101
            - qdw * qdz * s101
            + q0w * q0z * (-s001 + s101)
            + q0x * q0x * (s011 - s111)
            - q0z * q0z * s111
            + qdx * qdx * s111
            + qdz * qdz * s111
            + q0x * (q0y * (-s001 + s101) + q0w * (s021 - s121))
            + q0y * q0z * s121
            + qdw * qdx * s121
            - qdy * qdz * s121)
            * theta,
        -2.0 * (qdx * qdy * s002 + qdw * qdz * s002 + q0z * q0z * s012
            - qdx * qdx * s012
            - qdz * qdz * s012
            - q0y * q0z * s022
            - qdw * qdx * s022
            + qdy * qdz * s022
            - qdx * qdy * s102
            - qdw * qdz * s102
            + q0w * q0z * (-s002 + s102)
            + q0x * q0x * (s012 - s112)
            - q0z * q0z * s112
            + qdx * qdx * s112
            + qdz * qdz * s112
            + q0x * (q0y * (-s002 + s102) + q0w * (s022 - s122))
            + q0y * q0z * s122
            + qdw * qdx * s122
            - qdy * qdz * s122)
            * theta,
    );

    let c1_2 = DerivativeTerm::new(
        -t0z + t1z,
        qdw * qdy * s000
            - qdx * qdz * s000
            - q0y * q0z * s010
            - qdw * qdx * s010
            - qdy * qdz * s010
            - s020
            + q0y * q0y * s020
            + qdx * qdx * s020
            + qdy * qdy * s020
            - qdw * qdy * s100
            + qdx * qdz * s100
            + q0x * q0z * (-s000 + s100)
            + q0y * q0z * s110
            + qdw * qdx * s110
            + qdy * qdz * s110
            + q0w * (q0y * (s000 - s100) + q0x * (-s010 + s110))
            + q0x * q0x * (s020 - s120)
            + s120
            - q0y * q0y * s120
            - qdx * qdx * s120
            - qdy * qdy * s120,
        qdw * qdy * s001
            - qdx * qdz * s001
            - q0y * q0z * s011
            - qdw * qdx * s011
            - qdy * qdz * s011
            - s021
            + q0y * q0y * s021
            + qdx * qdx * s021
            + qdy * qdy * s021
            - qdw * qdy * s101
            + qdx * qdz * s101
            + q0x * q0z * (-s001 + s101)
            + q0y * q0z * s111
            + qdw * qdx * s111
            + qdy * qdz * s111
            + q0w * (q0y * (s001 - s101) + q0x * (-s011 + s111))
            + q0x * q0x * (s021 - s121)
            + s121
            - q0y * q0y * s121
            - qdx * qdx * s121
            - qdy * qdy * s121,
        qdw * qdy * s002
            - qdx * qdz * s002
            - q0y * q0z * s012
            - qdw * qdx * s012
            - qdy * qdz * s012
            - s022
            + q0y * q0y * s022
            + qdx * qdx * s022
            + qdy * qdy * s022
            - qdw * qdy * s102
            + qdx * qdz * s102
            + q0x * q0z * (-s002 + s102)
            + q0y * q0z * s112
            + qdw * qdx * s112
            + qdy * qdz * s112
            + q0w * (q0y * (s002 - s102) + q0x * (-s012 + s112))
            + q0x * q0x * (s022 - s122)
            + s122
            - q0y * q0y * s122
            - qdx * qdx * s122
            - qdy * qdy * s122,
    );

    let c2_2 = DerivativeTerm::new(
        0.0,
        q0w * q0y * s000 - q0x * q0z * s000 - qdw * qdy * s000 + qdx * qdz * s000
            - q0w * q0x * s010
            - q0y * q0z * s010
            + qdw * qdx * s010
            + qdy * qdz * s010
            + q0x * q0x * s020
            + q0y * q0y * s020
            - qdx * qdx * s020
            - qdy * qdy * s020
            - q0w * q0y * s100
            + q0x * q0z * s100
            + qdw * qdy * s100
            - qdx * qdz * s100
            + q0w * q0x * s110
            + q0y * q0z * s110
            - qdw * qdx * s110
            - qdy * qdz * s110
            - q0x * q0x * s120
            - q0y * q0y * s120
            + qdx * qdx * s120
            + qdy * qdy * s120
            - 2.0 * q0y * qdw * s000 * theta
            + 2.0 * q0z * qdx * s000 * theta
            - 2.0 * q0w * qdy * s000 * theta
            + 2.0 * q0x * qdz * s000 * theta
            + 2.0 * q0x * qdw * s010 * theta
            + 2.0 * q0w * qdx * s010 * theta
            + 2.0 * q0z * qdy * s010 * theta
            + 2.0 * q0y * qdz * s010 * theta
            - 4.0 * q0x * qdx * s020 * theta
            - 4.0 * q0y * qdy * s020 * theta,
        q0w * q0y * s001 - q0x * q0z * s001 - qdw * qdy * s001 + qdx * qdz * s001
            - q0w * q0x * s011
            - q0y * q0z * s011
            + qdw * qdx * s011
            + qdy * qdz * s011
            + q0x * q0x * s021
            + q0y * q0y * s021
            - qdx * qdx * s021
            - qdy * qdy * s021
            - q0w * q0y * s101
            + q0x * q0z * s101
            + qdw * qdy * s101
            - qdx * qdz * s101
            + q0w * q0x * s111
            + q0y * q0z * s111
            - qdw * qdx * s111
            - qdy * qdz * s111
            - q0x * q0x * s121
            - q0y * q0y * s121
            + qdx * qdx * s121
            + qdy * qdy * s121
            - 2.0 * q0y * qdw * s001 * theta
            + 2.0 * q0z * qdx * s001 * theta
            - 2.0 * q0w * qdy * s001 * theta
            + 2.0 * q0x * qdz * s001 * theta
            + 2.0 * q0x * qdw * s011 * theta
            + 2.0 * q0w * qdx * s011 * theta
            + 2.0 * q0z * qdy * s011 * theta
            + 2.0 * q0y * qdz * s011 * theta
            - 4.0 * q0x * qdx * s021 * theta
            - 4.0 * q0y * qdy * s021 * theta,
        q0w * q0y * s002 - q0x * q0z * s002 - qdw * qdy * s002 + qdx * qdz * s002
            - q0w * q0x * s012
            - q0y * q0z * s012
            + qdw * qdx * s012
            + qdy * qdz * s012
            + q0x * q0x * s022
            + q0y * q0y * s022
            - qdx * qdx * s022
            - qdy * qdy * s022
            - q0w * q0y * s102
            + q0x * q0z * s102
            + qdw * qdy * s102
            - qdx * qdz * s102
            + q0w * q0x * s112
            + q0y * q0z * s112
            - qdw * qdx * s112
            - qdy * qdz * s112
            - q0x * q0x * s122
            - q0y * q0y * s122
            + qdx * qdx * s122
            + qdy * qdy * s122
            - 2.0 * q0y * qdw * s002 * theta
            + 2.0 * q0z * qdx * s002 * theta
            - 2.0 * q0w * qdy * s002 * theta
            + 2.0 * q0x * qdz * s002 * theta
            + 2.0 * q0x * qdw * s012 * theta
            + 2.0 * q0w * qdx * s012 * theta
            + 2.0 * q0z * qdy * s012 * theta
            + 2.0 * q0y * qdz * s012 * theta
            - 4.0 * q0x * qdx * s022 * theta
            - 4.0 * q0y * qdy * s022 * theta,
    );

    let c3_2 = DerivativeTerm::new(
        0.0,
        -2.0 * (-(q0w * qdy * s000) + q0x * qdz * s000 + q0x * qdw * s010 + q0w * qdx * s010
            - 2.0 * q0x * qdx * s020
            + q0w * qdy * s100
            - q0x * qdz * s100
            - q0x * qdw * s110
            - q0w * qdx * s110
            + q0z * (qdx * s000 + qdy * s010 - qdx * s100 - qdy * s110)
            + 2.0 * q0x * qdx * s120
            + q0y
                * (qdz * s010 - 2.0 * qdy * s020 + qdw * (-s000 + s100) - qdz * s110
                    + 2.0 * qdy * s120))
            * theta,
        -2.0 * (-(q0w * qdy * s001) + q0x * qdz * s001 + q0x * qdw * s011 + q0w * qdx * s011
            - 2.0 * q0x * qdx * s021
            + q0w * qdy * s101
            - q0x * qdz * s101
            - q0x * qdw * s111
            - q0w * qdx * s111
            + q0z * (qdx * s001 + qdy * s011 - qdx * s101 - qdy * s111)
            + 2.0 * q0x * qdx * s121
            + q0y
                * (qdz * s011 - 2.0 * qdy * s021 + qdw * (-s001 + s101) - qdz * s111
                    + 2.0 * qdy * s121))
            * theta,
        -2.0 * (-(q0w * qdy * s002) + q0x * qdz * s002 + q0x * qdw * s012 + q0w * qdx * s012
            - 2.0 * q0x * qdx * s022
            + q0w * qdy * s102
            - q0x * qdz * s102
            - q0x * qdw * s112
            - q0w * qdx * s112
            + q0z * (qdx * s002 + qdy * s012 - qdx * s102 - qdy * s112)
            + 2.0 * q0x * qdx * s122
            + q0y
                * (qdz * s012 - 2.0 * qdy * s022 + qdw * (-s002 + s102) - qdz * s112
                    + 2.0 * qdy * s122))
            * theta,
    );

    let c4_2 = DerivativeTerm::new(
        0.0,
        q0w * qdy * s000 - q0x * qdz * s000 - q0x * qdw * s010 - q0w * qdx * s010
            + 2.0 * q0x * qdx * s020
            - q0w * qdy * s100
            + q0x * qdz * s100
            + q0x * qdw * s110
            + q0w * qdx * s110
            - 2.0 * q0x * qdx * s120
            - 2.0 * qdw * qdy * s000 * theta
            + 2.0 * qdx * qdz * s000 * theta
            - 2.0 * q0w * q0x * s010 * theta
            + 2.0 * qdw * qdx * s010 * theta
            + 2.0 * qdy * qdz * s010 * theta
            + 2.0 * q0x * q0x * s020 * theta
            + 2.0 * q0y * q0y * s020 * theta
            - 2.0 * qdx * qdx * s020 * theta
            - 2.0 * qdy * qdy * s020 * theta
            + q0z
                * (-(qdx * s000) - qdy * s010 + qdx * s100 + qdy * s110 - 2.0 * q0x * s000 * theta)
            + q0y
                * (-(qdz * s010) + 2.0 * qdy * s020 + qdw * (s000 - s100) + qdz * s110
                    - 2.0 * qdy * s120
                    + 2.0 * q0w * s000 * theta
                    - 2.0 * q0z * s010 * theta),
        q0w * qdy * s001 - q0x * qdz * s001 - q0x * qdw * s011 - q0w * qdx * s011
            + 2.0 * q0x * qdx * s021
            - q0w * qdy * s101
            + q0x * qdz * s101
            + q0x * qdw * s111
            + q0w * qdx * s111
            - 2.0 * q0x * qdx * s121
            - 2.0 * qdw * qdy * s001 * theta
            + 2.0 * qdx * qdz * s001 * theta
            - 2.0 * q0w * q0x * s011 * theta
            + 2.0 * qdw * qdx * s011 * theta
            + 2.0 * qdy * qdz * s011 * theta
            + 2.0 * q0x * q0x * s021 * theta
            + 2.0 * q0y * q0y * s021 * theta
            - 2.0 * qdx * qdx * s021 * theta
            - 2.0 * qdy * qdy * s021 * theta
            + q0z
                * (-(qdx * s001) - qdy * s011 + qdx * s101 + qdy * s111 - 2.0 * q0x * s001 * theta)
            + q0y
                * (-(qdz * s011) + 2.0 * qdy * s021 + qdw * (s001 - s101) + qdz * s111
                    - 2.0 * qdy * s121
                    + 2.0 * q0w * s001 * theta
                    - 2.0 * q0z * s011 * theta),
        q0w * qdy * s002 - q0x * qdz * s002 - q0x * qdw * s012 - q0w * qdx * s012
            + 2.0 * q0x * qdx * s022
            - q0w * qdy * s102
            + q0x * qdz * s102
            + q0x * qdw * s112
            + q0w * qdx * s112
            - 2.0 * q0x * qdx * s122
            - 2.0 * qdw * qdy * s002 * theta
            + 2.0 * qdx * qdz * s002 * theta
            - 2.0 * q0w * q0x * s012 * theta
            + 2.0 * qdw * qdx * s012 * theta
            + 2.0 * qdy * qdz * s012 * theta
            + 2.0 * q0x * q0x * s022 * theta
            + 2.0 * q0y * q0y * s022 * theta
            - 2.0 * qdx * qdx * s022 * theta
            - 2.0 * qdy * qdy * s022 * theta
            + q0z
                * (-(qdx * s002) - qdy * s012 + qdx * s102 + qdy * s112 - 2.0 * q0x * s002 * theta)
            + q0y
                * (-(qdz * s012) + 2.0 * qdy * s022 + qdw * (s002 - s102) + qdz * s112
                    - 2.0 * qdy * s122
                    + 2.0 * q0w * s002 * theta
                    - 2.0 * q0z * s012 * theta),
    );

    let c5_2 = DerivativeTerm::new(
        0.0,
        2.0 * (qdw * qdy * s000 - qdx * qdz * s000 + q0y * q0z * s010
            - qdw * qdx * s010
            - qdy * qdz * s010
            - q0y * q0y * s020
            + qdx * qdx * s020
            + qdy * qdy * s020
            + q0x * q0z * (s000 - s100)
            - qdw * qdy * s100
            + qdx * qdz * s100
            + q0w * (q0y * (-s000 + s100) + q0x * (s010 - s110))
            - q0y * q0z * s110
            + qdw * qdx * s110
            + qdy * qdz * s110
            + q0y * q0y * s120
            - qdx * qdx * s120
            - qdy * qdy * s120
            + q0x * q0x * (-s020 + s120))
            * theta,
        2.0 * (qdw * qdy * s001 - qdx * qdz * s001 + q0y * q0z * s011
            - qdw * qdx * s011
            - qdy * qdz * s011
            - q0y * q0y * s021
            + qdx * qdx * s021
            + qdy * qdy * s021
            + q0x * q0z * (s001 - s101)
            - qdw * qdy * s101
            + qdx * qdz * s101
            + q0w * (q0y * (-s001 + s101) + q0x * (s011 - s111))
            - q0y * q0z * s111
            + qdw * qdx * s111
            + qdy * qdz * s111
            + q0y * q0y * s121
            - qdx * qdx * s121
            - qdy * qdy * s121
            + q0x * q0x * (-s021 + s121))
            * theta,
        2.0 * (qdw * qdy * s002 - qdx * qdz * s002 + q0y * q0z * s012
            - qdw * qdx * s012
            - qdy * qdz * s012
            - q0y * q0y * s022
            + qdx * qdx * s022
            + qdy * qdy * s022
            + q0x * q0z * (s002 - s102)
            - qdw * qdy * s102
            + qdx * qdz * s102
            + q0w * (q0y * (-s002 + s102) + q0x * (s012 - s112))
            - q0y * q0z * s112
            + qdw * qdx * s112
            + qdy * qdz * s112
            + q0y * q0y * s122
            - qdx * qdx * s122
            - qdy * qdy * s122
            + q0x * q0x * (-s022 + s122))
            * theta,
    );

    let c1 = [c1_0, c1_1, c1_2];
    let c2 = [c2_0, c2_1, c2_2];
    let c3 = [c3_0, c3_1, c3_2];
    let c4 = [c4_0, c4_1, c4_2];
    let c5 = [c5_0, c5_1, c5_2];
    return [c1, c2, c3, c4, c5];
}
