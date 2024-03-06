import re
import math
import string
from typing import Final


ARH_re: Final = r"(?:(?:\d+|[a-z])(?:[\+\-\*](?:\d+|[a-z]))*)"
ARG_re: Final = f"({ARH_re}(?:,{ARH_re})*)"
RHS_re: Final = f"(?:(T|D)\({ARG_re}\))"
LHS_re: Final = r"(?:(T)\(([a-z](?:,[a-z])*)\))"
XPR_re: Final = f"^{LHS_re}<-{RHS_re}(\*{RHS_re})*$"
SPECIAL: Final = set("TD,()<-*+*-0123456789")


def validate_expr(expr: str, bounds: dict[str, int]) -> None:
    expr = expr.replace(" ", "")
    used = set(expr) - SPECIAL
    assert re.match(XPR_re, expr), "Expression does not match the required regex"
    assert all([u in bounds for u in used]), "One of the indices is not present in `bounds` dict"
    assert all([b > 0 for b in bounds.items()]), "Bounds can only be >= 1"

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


def derive(expr: str) -> tuple[str, dict[str, int]]:
    expr = expr.replace(" ", "")

    lhs, rhs = expr.split("<-")
    lhs, rhs = re.findall(LHS_re, lhs)[0], re.findall(RHS_re, rhs)

    used_args = set(expr) - SPECIAL
    free_args = lhs[1].split(",")
    dumm_args = used_args - set(free_args)

    derivatives = []
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

        derivatives.append(("".join(args_y), derivative))

    return derivatives


def prog_cuda(expr: str, bounds: dict[str, int], shapes: list[tuple[int]]):
    expr = expr.replace(" ", "")

    lhs, rhs = expr.split("<-")
    lhs, rhs = re.findall(LHS_re, lhs)[0], re.findall(RHS_re, rhs)
    rhs_Ts = [args for ttype, args in rhs if ttype == "T"]
    rhs_Ds = [args for ttype, args in rhs if ttype == "D"]

    used_args = set(expr) - SPECIAL
    free_args = lhs[1].split(",")
    dumm_args = used_args - set(free_args)

    out_shape = get_output_shape(expr, bounds)
    size = math.prod(out_shape)

    f_signature = ", ".join(f"double *T{Tid}" for Tid in range(len(rhs_Ts)))
    free_args_vals = [
        f"(item % {math.prod(out_shape[:i])}) / {math.prod(out_shape[:i-1])}"
        for i in range(1, len(out_shape) + 1)
    ]
    free_args_init = "long "
    free_args_init += ", ".join(f"{arg} = {value}" for arg, value in zip(free_args, free_args_vals))
    for_loops = "\n".join(f"for(long {arg}=0; {arg}<{bounds[arg]}; {arg}++){{" for arg in dumm_args)
    for_end = "}" * len(dumm_args)
    linearize = lambda Tid: " + ".join(
        f"({arg})*{math.prod([shapes[Tid][ax_id] for ax_id in range(n)])}"
        for n, arg in enumerate(rhs_Ts[Tid].split(","))
    )
    ein_expr = " * ".join(
        [f"T{Tid}[{linearize(Tid)}]" for Tid in range(len(rhs_Ts))]
        + [f"(({args.split(',')[0]}=={args.split(',')[1]}) ? 1 : 0)" for args in rhs_Ds]
    )

    prog = f"""extern "C"
__global__ void ein({f_signature}, double *R){{
long item = (blockIdx.x * blockDim.x) + threadIdx.x;
if (item < {size}){{
{free_args_init};
double acc = 0;
{for_loops}
acc += {ein_expr};
{for_end}
R[item] = acc;
}}
}}
"""
    return prog


if __name__ == "__main__":
    expr = "T(a, b, c, e) <- T( 30*a + i, 3*b + j, k, 0 ) * T(i, j, k, c) * D(i,j)"
    bounds = dict(zip("abceijk", (3, 3, 3, 4, 4, 4, 4)))
    shapes = [(2, 2, 2, 2), (2, 2, 2, 2)]

    expr = "T(j) <- T(i,i)"
    bounds = dict(zip("ji", (1, 4)))
    shapes = [(4, 4)]

    # print(expr, "\n", "=" * 60)
    for d in derive(expr):
        print(d)

    # print(expr)
    # print(prog_cuda(expr, bounds, shapes))
