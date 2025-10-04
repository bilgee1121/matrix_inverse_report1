# Report 1: 역행렬 프로그램
# 작성자: 빌렉자르갈

from fractions import Fraction

def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def pretty_matrix(M, prec=6):
    return "\n".join("[ " + "  ".join(f"{x:.{prec}g}" for x in row) + " ]" for row in M)

def det_bareiss(Ain):
    A = [[Fraction(x) for x in row] for row in Ain]
    n = len(A)
    sign, denom = 1, Fraction(1)
    for k in range(n - 1):
        # pivot 찾기
        piv = next((i for i in range(k, n) if A[i][k] != 0), None)
        if piv is None:
            return 0
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            sign *= -1
        pivot = A[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * pivot - A[i][k] * A[k][j]) / (denom if denom != 0 else 1)
            A[i][k] = Fraction(0)
        denom = pivot
    det = sign * A[n - 1][n - 1]
    return int(det) if getattr(det, "denominator", 1) == 1 else float(det)

def minor_matrix(A, r, c):
    return [row[:c] + row[c + 1 :] for i, row in enumerate(A) if i != r]

def adjugate_inverse(A):
    n = len(A)
    detA = det_bareiss(A)
    if detA == 0:
        raise ZeroDivisionError("역행렬 없음 (det=0)")
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = ((-1) ** (i + j)) * det_bareiss(minor_matrix(A, i, j))
    adj = [[C[j][i] for j in range(n)] for i in range(n)]
    return [[adj[i][j] / detA for j in range(n)] for i in range(n)]

def gauss_jordan_inverse(Ain, eps=1e-12):
    n = len(Ain)
    A = [[float(x) for x in row] for row in Ain]
    I = identity(n)
    for i in range(n):
        A[i] += I[i]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot_row][col]) < eps:
            raise ZeroDivisionError("역행렬 없음 (pivot=0)")
        if pivot_row != col:
            A[col], A[pivot_row] = A[pivot_row], A[col]
        piv = A[col][col]
        A[col] = [x / piv for x in A[col]]
        for r in range(n):
            if r == col:
                continue
            factor = A[r][col]
            A[r] = [A[r][c] - factor * A[col][c] for c in range(2 * n)]
    return [row[n:] for row in A]

def matrices_close(A, B, tol=1e-9):
    return all(abs(A[i][j] - B[i][j]) <= tol for i in range(len(A)) for j in range(len(A)))

def main():
    print("=== Report 1: 행렬 역행렬 ===")
    n = int(input("정수 n 입력: "))
    print(f"{n}×{n} 행렬 입력 (공백 구분)")
    A = [list(map(float, input().split())) for _ in range(n)]
    print("\n입력 행렬 A:")
    print(pretty_matrix(A))

    try:
        inv1 = adjugate_inverse(A)
        print("\n[방법1] 행렬식 기반 역행렬:")
        print(pretty_matrix(inv1))
    except Exception as e:
        print("[방법1] 실패:", e)
        inv1 = None

    try:
        inv2 = gauss_jordan_inverse(A)
        print("\n[방법2] 가우스-조던 기반 역행렬:")
        print(pretty_matrix(inv2))
    except Exception as e:
        print("[방법2] 실패:", e)
        inv2 = None

    if inv1 and inv2:
        print("\n두 결과 동일 여부:", matrices_close(inv1, inv2))
    else:
        print("\n비교 불가")

if __name__ == "__main__":
    main()
