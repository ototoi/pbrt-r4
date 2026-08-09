use pbrt_r4::util::lowdiscrepancy::radical_inverse::reverse_bits64;

#[test]
fn reverse_bits64_is_involutive() {
    let u1 = 1;
    let u2 = reverse_bits64(u1);
    let u3 = reverse_bits64(u2);
    assert_eq!(u1, u3);
}
