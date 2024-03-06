import re
import math
import string
from typing import Final


ARH_re: Final = r"(?:(?:\d+|[a-z])(?:[\+\-\*](?:\d+|[a-z]))*)"
ARG_re: Final = f"({ARH_re}(?:,{ARH_re})*)"
RHS_re: Final = f"(?:(T|D)\({ARG_re}\))"
LHS_re: Final = r"(?:(T)\(([a-z](?:,[a-z])*)\))"
XPR_re: Final = f"^{LHS_re}<-{RHS_re}(\*{RHS_re})*$"
SPECIAL: Final = {"T", "D"} + {",", "(", ")", "<-", "*"} + {"+", "*", "-"} + set("0123456789")


def validate_expr(expr: str, bounds: dict[str, int]) -> None:
    expr = expr.replace(" ", "")
    used = set(expr) - SPECIAL
    assert re.match(XPR_re, expr), "Expression does not match the required regex"
    assert all([u in bounds for u in used]), "One of the indices is not present in `bounds` dict"
    assert all([b > 0 for b in bounds.items()]), "One of the bounds is 0 (can only be >= 1)"

    # Make sure that every Delta expression has exactly 2 args
    lhs, rhs = expr.split("<-")
    lhs, rhs = re.findall(LHS_re, lhs)[0], re.findall(RHS_re, rhs)
    assert all([ttype != "D" or len(args.split(",")) == 2 for ttype, args in rhs])


def make_indices(n: int, used_indices: set[str] = set()) -> list[str]:
    assert len(string.ascii_lowercase) - len(used_indices) >= n
    return list(set(string.ascii_lowercase) - used_indices)[:n]


def delta_prod(indices1: list[str], indices2: list[str]) -> str:
    assert len(indices1) == len(indices2)
    return " * ".join(f"D({i1},{i2})" for i1, i2 in zip(indices1, indices2))


def get_output_shape(expr: str, bounds: dict[str, int]) -> tuple[int]:
    expr = expr.replace(" ", "")
    lhs, rhs = expr.split("<-")
    lhs, rhs = re.findall(LHS_re, lhs)[0], re.findall(RHS_re, rhs)
    return tuple(bounds[i] for i in lhs[1].split(","))


def derive(
    expr: str, bounds: dict[str, int], dv_bounds: dict[str, int]
) -> tuple[str, dict[str, int]]:
    expr = expr.replace(" ", "")

    lhs, rhs = expr.split("<-")
    lhs, rhs = re.findall(LHS_re, lhs)[0], re.findall(RHS_re, rhs)

    used_args = set(expr) - SPECIAL
    free_args = lhs[1].split(",")
    dumm_args = used_args - set(free_args)

    derivatives, new_bounds = [], []
    for n, (ttype, args_x) in enumerate(rhs):
        if ttype != "T":
            continue

        args_x = args_x.split(",")
        args_y = make_indices(len(args_x), used_args)

        derivative = " * ".join(f"{ttype[0]}({args_z})" for ttype, args_z in rhs[:n])
        derivative += " * " if rhs[:n] else ""
        derivative += delta_prod(args_y, args_x)
        derivative += " * " if rhs[n + 1 :] else ""
        derivative += " * ".join(f"{ttype[0]}({args_z})" for ttype, args_z in rhs[n + 1 :])
        derivative = f"T({','.join(args_y)},{','.join(free_args)}) <- {derivative}"

        for x, y in zip(args_x, args_y):
            derivative = derivative.replace(x, y) if x in dumm_args else derivative

        derivatives.append(derivative)
        new_bounds.append({**{y: dv_bounds[i] for i, y in enumerate(args_y)}, **bounds})

    return list(zip(derivatives, new_bounds))


def prog_cuda(expr: str, bounds: dict[str, int], shapes: list[tuple[int]]):
    expr = expr.replace(" ", "")

    lhs, rhs = expr.split("<-")
    lhs, rhs = re.findall(LHS_re, lhs)[0], re.findall(RHS_re, rhs)

    used_args = set(expr) - SPECIAL
    free_args = lhs[1].split(",")
    dumm_args = used_args - set(free_args)

    out_shape = get_output_shape(expr, bounds)

    func_signature = ", ".join(f"double *T{id}" for id in range(len(rhs)))
    free_args_vals = [
        f"(item % {math.prod(out_shape[:i])}) / {math.prod(out_shape[:i-1])}"
        for i in range(1, len(out_shape) + 1)
    ]
    free_args_init = "long "
    free_args_init += ", ".join(f"{arg} = {value}" for arg, value in zip(free_args, free_args_vals))
    for_loops = "\n".join(f"for(long {arg}=0; {arg}<{bounds[arg]}; {arg}++){{" for arg in dumm_args)
    for_end = "}" * len(dumm_args)

    linearize_arg = lambda Tid: "+".join()


if __name__ == "__main__":
    expr = "T(a, b, c, e) <- T( 30*a + i, 3*b + j, k, 0 ) * T(i, j, k, c) * D(i,j)"
    # expr = "T(j) <- T(i,i,i,i)"

    print(expr, "\n", "=" * 60)
    for d, _ in derive(expr, dict(zip("abceijk", (3, 3, 3, 4, 4, 4, 4)))):
        print(d)
