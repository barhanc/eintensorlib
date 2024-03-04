from string import ascii_letters


def derive(expr: str) -> str:
    pass


def makeIndices(n: int, usedIndices: set[str] = set()) -> list[str]:
    return list(set(ascii_letters) - usedIndices)[:n]


def deltaProd(indices1: list[str], indices2: list[str]) -> str:
    return "*".join(f"δ({i1},{i2})" for i1, i2 in zip(indices1, indices2))


def getBounds(expr: str, *Ts) -> dict[str, int]:
    assert "+" not in expr
    expr = list(filter(lambda s: s[0] == "T", expr.replace(" ", "").split("*")))
    bounds = {}
    for i, T in enumerate(Ts):
        for j, idx in enumerate(expr[i][2:-1].split(",")):
            if idx in bounds:
                assert bounds[idx] == T.shape[j]
            else:
                bounds[idx] = T.shape[j]
    return bounds


if __name__ == "__main__":
    import numpy as np

    print(getBounds("T(i, j, k) * T(i, j, q) * δ(p, q)  ", np.ones((3, 3, 3)), np.ones((3, 3, 3))))
