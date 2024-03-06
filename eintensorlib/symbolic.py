import re
from string import ascii_lowercase


lT = r"([TD]\((|[a-z]?+(?:,[a-z])*)\))"
rT = r"([TD]\((|(?:(?:\d+|[a-z])(?:[\+\-\*](?:\d+|[a-z]))*)?+(?:,(?:(?:\d+|[a-z])(?:[\+\-\*](?:\d+|[a-z]))*)?)*)\))"
aT = f"{lT}<-{rT}(\*{rT})*"


def derive(expr: str) -> str:
    expr = expr.replace(" ", "")
    lhs, rhs = expr.split("<-")

    used = set(expr) - {"T", "D", ",", "(", ")", "<", "-", "+", "*"}
    free = get_indices_T(lhs)
    dumm = used - set(free)

    Texprs, derivatives = rhs.split(")*T"), []
    for i, T in enumerate(Texprs):
        if T[0] != "T":
            continue

        a = get_indices_T(T)
        b = make_indices(len(a), used)

        derivative = (
            "*".join(Texprs[:i])
            + ("*" if Texprs[:i] else "")
            + delta_prod(b, a)
            + ("*" if Texprs[i + 1 :] else "")
            + "*".join(Texprs[i + 1 :])
        )
        derivative = (
            f"T({','.join(b)}{',' if len(free[0]) else ''}{','.join(free)}) <- " + derivative
        )

        for ia, ib in zip(a, b):
            derivative = derivative.replace(ia, ib) if ia in dumm else derivative

        derivatives.append(derivative)

    return derivatives


def make_indices(n: int, used_indices: set[str] = set()) -> list[str]:
    return list(set(ascii_lowercase) - used_indices)[:n]


def delta_prod(indices1: list[str], indices2: list[str]) -> str:
    assert len(indices1) == len(indices2)
    return "*".join(f"D({i1},{i2})" for i1, i2 in zip(indices1, indices2))


def get_indices_T(expr: str):
    expr = expr.replace(" ", "").replace("(", "").replace(")", "")
    expr = expr.replace("T", "").replace("D", "")
    return expr.split(",")


import re

if __name__ == "__main__":
    # print(derive("T(i,j) <- T(i,k) * T(j,k)"))
    # print(derive("T( ) <- T(i,i,i,i)"))
    # print(derive("T(a,b,c,e) <- T(3*a+i, 3*b+j, k, e)*T(i,j,k,c)"))

    expr = "T(a,b,c,e)<-T(3*a+i,3*b+j,k,e)*T(i,j,k,c)"
    assert re.match(aT, expr)
    lhs, rhs = expr.split("<-")
    print(re.findall(lT, lhs))
    print(re.findall(rT, rhs))
