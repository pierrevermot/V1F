import numpy as np


def generate_elt_pupil(
    n_pix: int,
    *,
    diameter_m: float = 40.0,
    spider_width_m: float = 0.51,
    gap_m: float = 4e-3,
    rotation_deg: float = 0.0,
    central_obscuration_ratio: float = 0.0,
    reflectivity_std: float | None = None,
    missing_segments: int = 0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return a float32 array containing the ELT pupil mask.

    Parameters mirror the COMPASS ParamTel settings:
      * n_pix: number of pixels across the pupil support (use p_geom.pupdiam).
      * diameter_m: primary mirror diameter in metres.
      * spider_width_m: width of the three spiders, metres.
      * gap_m: inter-segment gap, metres.
      * rotation_deg: pupil rotation angle, degrees.
      * central_obscuration_ratio: central obscuration diameter ratio (0–1).
      * reflectivity_std: optional per-segment reflectivity std dev (metres).
      * missing_segments: number of random missing segments (set to 0 for full pupil).
      * rng: optional numpy RNG or seed for reproducibility (defaults to 42).
    """
    if n_pix <= 0:
        raise ValueError("n_pix must be positive")

    if rng is None:
        rng = np.random.default_rng(42)
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    def _dist(N, xc, yc):
        x = np.arange(N, dtype=np.float64) - xc
        y = np.arange(N, dtype=np.float64) - yc
        X, Y = np.meshgrid(x, y, indexing="ij")
        return np.sqrt(X**2 + Y**2)

    def _fill_polygon(x, y, i0, j0, scale, gap, N, index=0):
        X = (np.arange(N, dtype=np.float64) - i0) * scale
        Y = (np.arange(N, dtype=np.float64) - j0) * scale
        X, Y = np.meshgrid(X, Y, indexing="ij")
        x0 = np.mean(x)
        y0 = np.mean(y)
        T = (np.arctan2(Y - y0, X - x0) + 2 * np.pi) % (2 * np.pi)
        t = (np.arctan2(y - y0, x - x0) + 2 * np.pi) % (2 * np.pi)
        sens = np.median(np.diff(np.unwrap(t)))
        if sens < 0:
            x = x[::-1]
            y = y[::-1]
            t = t[::-1]
        imin = t.argmin()
        if imin != 0:
            x = np.roll(x, -imin)
            y = np.roll(y, -imin)
            t = np.roll(t, -imin)
        n = x.shape[0]
        indx = np.array([], dtype=np.int64)
        indy = np.array([], dtype=np.int64)
        distedge = np.array([], dtype=np.float64)
        for i in range(n):
            j = (i + 1) % n
            if j == 0:
                sub = np.where((T >= t[-1]) | (T <= t[0]))
            else:
                sub = np.where((T >= t[i]) & (T <= t[j]))
            dy = y[j] - y[i]
            dx = x[j] - x[i]
            vnorm = np.hypot(dx, dy)
            if vnorm == 0:
                continue
            dx /= vnorm
            dy /= vnorm
            crossprod = dx * (Y[sub] - y[i]) - dy * (X[sub] - x[i])
            tmp = crossprod > gap
            indx = np.append(indx, sub[0][tmp])
            indy = np.append(indy, sub[1][tmp])
            distedge = np.append(distedge, crossprod[tmp])
        if index == 1:
            return indx.astype(np.int64), indy.astype(np.int64), distedge
        pol = np.zeros((N, N), dtype=bool)
        pol[indx, indy] = True
        return pol

    def _fill_spider(N, nspider, dspider, i0, j0, scale, rot):
        mask = np.ones((N, N), dtype=bool)
        x = (np.arange(N, dtype=np.float64) - i0) * scale
        y = (np.arange(N, dtype=np.float64) - j0) * scale
        X, Y = np.meshgrid(x, y, indexing="ij")
        w = 2 * np.pi / nspider
        for i in range(nspider):
            nn = np.abs(X * np.cos(i * w - rot) + Y * np.sin(i * w - rot)) < dspider / 2.0
            mask[nn] = False
        return mask

    def _create_hexa_pattern(pitch, support_size):
        v3 = np.sqrt(3.0)
        nx = int(np.ceil((support_size / 2.0) / pitch) + 1)
        x = pitch * (np.arange(2 * nx + 1) - nx)
        ny = int(np.ceil((support_size / 2.0) / pitch / v3) + 1)
        y = (v3 * pitch) * (np.arange(2 * ny + 1) - ny)
        x, y = np.meshgrid(x, y, indexing="ij")
        x = x.flatten()
        y = y.flatten()
        peak_axis = np.append(x, x + pitch / 2.0)
        flat_axis = np.append(y, y + pitch * v3 / 2.0)
        return flat_axis, peak_axis

    def _reorganize_segments_order_eso(x, y):
        pi_3 = np.pi / 3
        pi_6 = np.pi / 6
        two_pi = 2 * np.pi
        t = (np.arctan2(y, x) + pi_6 - 1e-3) % two_pi
        X = np.array([])
        Y = np.array([])
        A = 100.0
        for k in range(6):
            sector = (t > k * pi_3) & (t < (k + 1) * pi_3)
            u = k * pi_3
            distance = (A * np.cos(u) - np.sin(u)) * x[sector] + (np.cos(u) + A * np.sin(u)) * y[sector]
            indsort = np.argsort(distance)
            X = np.append(X, x[sector][indsort])
            Y = np.append(Y, y[sector][indsort])
        return X, Y

    def _generate_coord_segments(
        D,
        rot,
        pitch=1.244683637214,
        nseg=33,
        inner_rad=4.1,
        outer_rad=15.4,
        R=95.7853,
        nominalD=40,
    ):
        v3 = np.sqrt(3.0)
        lx, ly = _create_hexa_pattern(pitch, (nseg + 2) * pitch)
        ll = np.sqrt(lx**2 + ly**2)
        valid = (ll > inner_rad * pitch) & (ll < outer_rad * pitch)
        lx = lx[valid]
        ly = ly[valid]
        lx, ly = _reorganize_segments_order_eso(lx, ly)
        th = np.linspace(0, 2 * np.pi, 7)[:-1]
        hx = np.cos(th) * pitch / v3
        hy = np.sin(th) * pitch / v3
        x = lx[None, :] + hx[:, None]
        y = ly[None, :] + hy[:, None]
        r = np.sqrt(x**2 + y**2)
        rrc = R / r * np.arctan(r / R)
        x *= rrc
        y *= rrc
        if D != nominalD:
            scale = D / nominalD
            x *= scale
            y *= scale
        mrot = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
        xyrot = np.dot(mrot, np.transpose(np.array([x, y]), (1, 0, 2)))
        return xyrot[0], xyrot[1]

    def _generate_segment_properties(
        attribute,
        hx,
        hy,
        i0,
        j0,
        scale,
        gap,
        N,
        D,
        nominalD=40,
        pitch=1.244683637214,
        half_seg=0.75,
    ):
        nseg = hx.shape[-1]
        attr = np.asarray(attribute, dtype=np.float64)
        if attr.ndim == 0:
            attr = np.full(nseg, attr, dtype=np.float64)
        if attr.size != nseg:
            raise ValueError(f"attribute must have length {nseg}, got {attr.size}")
        pupil = np.zeros((N, N), dtype=np.float64)
        x0 = np.mean(hx, axis=0)
        y0 = np.mean(hy, axis=0)
        x0 = x0 / scale + i0
        y0 = y0 / scale + j0
        hexrad = half_seg * D / nominalD / scale
        ix0 = np.floor(x0 - hexrad).astype(int) - 1
        iy0 = np.floor(y0 - hexrad).astype(int) - 1
        segdiam = int(np.ceil(hexrad * 2 + 1)) + 1
        for idx in range(nseg):
            subx, suby, _ = _fill_polygon(
                hx[:, idx],
                hy[:, idx],
                i0 - ix0[idx],
                j0 - iy0[idx],
                scale,
                gap,
                segdiam,
                index=1,
            )
            sx = subx + ix0[idx]
            sy = suby + iy0[idx]
            valid = (sx >= 0) & (sx < N) & (sy >= 0) & (sy < N)
            pupil[sx[valid], sy[valid]] = attr[idx]
        return pupil

    pixscale = diameter_m / n_pix
    centre = n_pix / 2.0 - 0.5
    hx, hy = _generate_coord_segments(
        diameter_m,
        np.deg2rad(rotation_deg),
        pitch=1.244683637214,
        nseg=33,
        inner_rad=4.1,
        outer_rad=15.4,
        R=95.7853,
        nominalD=40,
    )
    nseg = hx.shape[-1]
    attribute = np.ones(nseg, dtype=np.float64)
    if reflectivity_std is not None and reflectivity_std > 0:
        attribute -= rng.normal(scale=reflectivity_std, size=nseg)
        attribute = np.clip(attribute, 0.0, None)
    if missing_segments:
        missing_segments = int(min(max(missing_segments, 0), nseg))
        if missing_segments > 0:
            idx = rng.choice(nseg, size=missing_segments, replace=False)
            attribute[idx] = 0.0
    pupil = _generate_segment_properties(
        attribute,
        hx,
        hy,
        centre,
        centre,
        pixscale,
        gap_m,
        n_pix,
        diameter_m,
        nominalD=40,
        pitch=1.244683637214,
        half_seg=0.75,
    )
    if spider_width_m > 0:
        spider_mask = _fill_spider(n_pix, 3, spider_width_m, centre, centre, pixscale, np.deg2rad(rotation_deg))
        pupil *= spider_mask
    if central_obscuration_ratio > 0:
        radius_mask = _dist(n_pix, centre, centre) >= (n_pix * central_obscuration_ratio + 1.0) * 0.5
        pupil *= radius_mask
    return pupil.astype(np.float32)
