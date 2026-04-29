from dataclasses import dataclass


@dataclass(frozen=True)
class MixedOrbital:
    coarse: object
    patch_values: dict


def add_mixed_orbitals(a, b):
    patch_values = {}
    patch_keys = set(a.patch_values) | set(b.patch_values)
    for key in patch_keys:
        a_val = a.patch_values.get(key)
        b_val = b.patch_values.get(key)
        if a_val is None:
            patch_values[key] = b_val
        elif b_val is None:
            patch_values[key] = a_val
        else:
            patch_values[key] = a_val + b_val
    return MixedOrbital(coarse=a.coarse + b.coarse, patch_values=patch_values)


def scale_mixed_orbital(alpha, orbital):
    return MixedOrbital(
        coarse=alpha * orbital.coarse,
        patch_values={key: alpha * value for key, value in orbital.patch_values.items()},
    )
