# Issue #128 governed baseline installation v1

On 2026-08-31, Ralph Kuepper explicitly approved all nine images in the
`bloom-quality-baseline-review-v1` bundle generated from renderer revision
`09ad0b755af9f10083712327d7f0edb1d88f228b`.

The reviewed qualification result has SHA-256
`49de2b4a46d2f4ad309908777769c0de81d182a9f403afb294de59224cf24d37`.
The review JSON has SHA-256
`883fee9b552cb27b8eb28921b935bedc0e4d42b4f1359998b0aa7b553cfce0ed`,
and its corpus manifest has SHA-256
`deee7f2528525c492146fc97c1a52551dad5dea5b5a8ffe4e1ef2c61158333e1`.

The governed installer reported `approved_by: Ralph Kuepper` and wrote a
`bloom-quality-baseline-install-v1` receipt with SHA-256
`82a99b96ab7e402d9dadd554e31e3850e61b6e30ed7394236a8289158bf01ab9`.
All nine targets were absent before installation. After installation, each
target was independently hashed and matched both its reviewed `after.png` and
the receipt exactly.

| Case | Installed SHA-256 |
|---|---|
| PBR spheres high | `e71f00d571e22023bf54ee8fe343964ed4877066ca874e52b2b94e31cf53c5a4` |
| PBR spheres constrained | `61f209370e715b815b32cf67e11527c717830c09dc8616c50ebe828640510c98` |
| Damaged Helmet | `3ca1d6c1dc1304513e765b9391188ca7fd9c0c2cb41b61e5664b310cd4eccf25` |
| Sponza interior | `32837d3da922647db87b59b4c9fbd2854a79417b28bdac33e8267f0599d9177b` |
| Bistro exterior | `9274c04b74cccd71c2dbb1c9d47cf54fac57a106b660f40e49650ce1e97bbb66` |
| Skinned alpha motion | `18ccdef6405e764c9c49c13e89aa844e0d2a7185db315abd90fadb1a3dd10396` |
| Draw/light stress | `5f23f5dd2b239941052f2c1fedddb9c5d470d5e914fda8f857d4e6072a0a0862` |
| Weighted transparency | `094172fca31801f19ce7ad1ab6ae01b9d89f5c6b3a8459bed54166e18b7cbaf5` |
| Masked alpha coverage | `04b25e87ddf06e5c911490417035180d1d26934f3a05449dc0ace23e24f40339` |

These files establish the first portable, human-approved Bloom renderer
quality corpus. A clean strict run from the committed installation is required
before closing the Metal qualification gate.
