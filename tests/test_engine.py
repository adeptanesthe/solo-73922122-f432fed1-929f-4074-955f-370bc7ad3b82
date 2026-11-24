"""核心引擎测试：必读坐标裁剪、稀疏求值对照、a=0、边角。"""

import numpy as np
import pytest

from app import engine, pack as pack_mod


def make_pack(C=2, O=3, seed=0, zero_a_channels=()):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-4, 4, C)
    for c in zero_a_channels:
        a[c] = 0.0
    b = rng.uniform(-3, 3, C)
    K = rng.uniform(-2, 2, (O, C, 3, 3))
    bias = rng.uniform(-1, 1, O)
    return pack_mod.build_pack(a, b, K, bias)


def pixels_from_image(image, coords):
    return {tuple(rc): image[rc[0], rc[1]].tolist() for rc in coords}


def test_pack_shape_independent_of_image_and_window():
    pack = make_pack(C=4, O=4, seed=1)
    token_small = pack_mod.encode_pack(pack)
    # 同一参数包用于完全不同的图/窗口尺寸
    for H, W, win in [(3, 3, (0, 0, 1, 1)), (40, 7, (30, 2, 40, 7)),
                      (1, 1, (0, 0, 1, 1)), (5, 2, (0, 0, 5, 2))]:
        rng = np.random.default_rng(H * W)
        img = rng.uniform(-16, 16, (H, W, 4))
        req = engine.required_coordinates(pack, H, W, *win)
        px = pixels_from_image(img, req["coords"])
        out = engine.evaluate_window(pack, H, W, *win, px)
        ref = engine.reference_full_image(pack, img)
        r0, c0, r1, c1 = win
        assert out.shape == (r1 - r0, c1 - c0, 4)
        np.testing.assert_allclose(out, ref[r0:r1, c0:c1], atol=1e-12, rtol=0)
    assert pack_mod.encode_pack(pack) == token_small


@pytest.mark.parametrize("seed", range(12))
def test_random_match_reference_including_corners(seed):
    rng = np.random.default_rng(100 + seed)
    C, O = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    H, W = int(rng.integers(1, 9)), int(rng.integers(1, 9))
    pack = make_pack(C=C, O=O, seed=seed,
                     zero_a_channels=(0,) if C >= 2 and seed % 3 == 0 else ())
    img = rng.uniform(-16, 16, (H, W, C))
    ref = engine.reference_full_image(pack, img)

    # 窗口覆盖：边角、单行、单列、整图、内部小块
    windows = [
        (0, 0, 1, 1),
        (0, 0, H, 1),
        (0, 0, 1, W),
        (0, 0, H, W),
        (H - 1, W - 1, H, W),
    ]
    if H >= 3 and W >= 3:
        windows.append((1, 1, H - 1, W - 1))
    for win in windows:
        req = engine.required_coordinates(pack, H, W, *win)
        out = engine.evaluate_window(
            pack, H, W, *win, pixels_from_image(img, req["coords"])
        )
        r0, c0, r1, c1 = win
        np.testing.assert_allclose(
            out, ref[r0:r1, c0:c1], atol=1e-10, rtol=0,
            err_msg=f"seed={seed} win={win}",
        )
        assert np.max(np.abs(out - ref[r0:r1, c0:c1])) <= 1e-8


def test_zero_a_channel_never_required_and_still_correct():
    pack = make_pack(C=3, O=2, seed=7, zero_a_channels=(1,))
    H = W = 6
    rng = np.random.default_rng(5)
    img = rng.uniform(-16, 16, (H, W, 3))
    req = engine.required_coordinates(pack, H, W, 0, 0, H, W)
    assert 1 not in req["active_channels"]
    # 非活跃通道（a=0）不读取任何像素，不列入 by_channel
    assert "1" not in req["by_channel"]
    # 提供的像素即使不带通道1信息（置零）也不影响结果
    px = {tuple(rc): [img[rc[0], rc[1], 0], 0.0, img[rc[0], rc[1], 2]]
          for rc in req["coords"]}
    out = engine.evaluate_window(pack, H, W, 0, 0, H, W, px)
    ref = engine.reference_full_image(pack, img)
    np.testing.assert_allclose(out, ref, atol=1e-12, rtol=0)


def test_zero_kernel_taps_not_required():
    C = O = 1
    pack = pack_mod.build_pack(
        a=[2.0], b=[1.0],
        kernel=np.zeros((1, 1, 3, 3)).tolist(),
        bias=[0.5],
    )
    req = engine.required_coordinates(pack, 5, 5, 0, 0, 5, 5)
    assert req["coords"] == []  # 全零卷积：无任何必读像素
    out = engine.evaluate_window(pack, 5, 5, 0, 0, 5, 5, pixels={})
    np.testing.assert_allclose(out, np.full((5, 5, 1), 0.5))


def test_sparse_tap_only_reads_math_dependent_pixels():
    # 只有中心 tap 非零：整图窗口的必读坐标应恰好是窗口本身（不向四周扩）
    K = np.zeros((1, 1, 3, 3))
    K[0, 0, 1, 1] = 1.5
    pack = pack_mod.build_pack(a=[2.0], b=[-1.0], kernel=K.tolist(), bias=[0.25])
    req = engine.required_coordinates(pack, 6, 6, 2, 1, 5, 4)
    expected = {(r, c) for r in range(2, 5) for c in range(1, 4)}
    assert set(map(tuple, req["coords"])) == expected


def test_missing_required_pixel_raises():
    pack = make_pack(C=2, O=2, seed=3)
    req = engine.required_coordinates(pack, 5, 5, 0, 0, 3, 3)
    px = pixels_from_image(np.zeros((5, 5, 2)), req["coords"])
    key = next(iter(px))
    del px[key]
    with pytest.raises(engine.EngineError, match="缺少必读像素"):
        engine.evaluate_window(pack, 5, 5, 0, 0, 3, 3, px)


def test_extra_pixels_are_allowed_but_not_needed():
    pack = make_pack(C=1, O=1, seed=11)
    H = W = 4
    rng = np.random.default_rng(2)
    img = rng.uniform(-16, 16, (H, W, 1))
    req = engine.required_coordinates(pack, H, W, 0, 0, 2, 2)
    px = pixels_from_image(img, req["coords"])
    px[(3, 3)] = img[3, 3].tolist()  # 窗口外的额外像素
    out = engine.evaluate_window(pack, H, W, 0, 0, 2, 2, px)
    ref = engine.reference_full_image(pack, img)
    np.testing.assert_allclose(out, ref[0:2, 0:2], atol=1e-12, rtol=0)


def test_all_channels_a_zero_still_emits_b_constant():
    # a 全为 0：不读任何像素，但 Y=b 仍经 K 产生输出（含边角补零差异）
    pack = pack_mod.build_pack(
        a=[0.0, 0.0], b=[2.0, -1.0],
        kernel=np.ones((1, 2, 3, 3)).tolist(), bias=[0.5],
    )
    H = W = 3
    img = np.zeros((H, W, 2))
    req = engine.required_coordinates(pack, H, W, 0, 0, H, W)
    assert req["coords"] == [] and req["active_channels"] == []
    out = engine.evaluate_window(pack, H, W, 0, 0, H, W, pixels={})
    ref = engine.reference_full_image(pack, img)
    np.testing.assert_allclose(out, ref, atol=1e-12, rtol=0)
    # 边角 4 个图内 tap、中心 9 个：常量 = bias + tap数 * Σ_c K*b
    assert out[0, 0, 0] == pytest.approx(0.5 + 4 * (2.0 - 1.0))
    assert out[1, 1, 0] == pytest.approx(0.5 + 9 * (2.0 - 1.0))


def test_invalid_window_and_shape_rejected():
    pack = make_pack()
    with pytest.raises(engine.EngineError):
        engine.required_coordinates(pack, 4, 4, 2, 2, 2, 4)  # r0 == r1
    with pytest.raises(engine.EngineError):
        engine.required_coordinates(pack, 4, 4, 0, 0, 5, 4)  # 越界
    with pytest.raises(engine.EngineError):
        engine.required_coordinates(pack, 0, 4, 0, 0, 1, 1)


def test_pack_constraints():
    with pytest.raises(pack_mod.PackError):
        pack_mod.build_pack(a=[1.0] * 5, b=[0.0] * 5,
                            kernel=np.zeros((1, 5, 3, 3)).tolist(), bias=[0.0])
    with pytest.raises(pack_mod.PackError):
        pack_mod.build_pack(a=[17.0], b=[0.0],
                            kernel=np.zeros((1, 1, 3, 3)).tolist(), bias=[0.0])
    with pytest.raises(pack_mod.PackError):
        pack_mod.build_pack(a=[1.0], b=[0.0],
                            kernel=np.zeros((1, 1, 2, 2)).tolist(), bias=[0.0])


def test_pack_roundtrip_preserves_arrays():
    pack = make_pack(C=3, O=4, seed=9)
    restored = pack_mod.decode_pack(pack_mod.encode_pack(pack))
    for k in ("a", "b", "K", "bias", "A", "t"):
        np.testing.assert_array_equal(restored[k], pack[k])
    with pytest.raises(pack_mod.PackError):
        pack_mod.decode_pack("not-base64!!!")
