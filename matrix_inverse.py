
# Report 1 : 역행렬을 구하는 프로그램
# 작성자 : 빌렉자르갈


from fractions import Fraction


def identity(n):
    """단위행렬 생성"""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def pretty_matrix(M, prec=6):
    """행렬을 보기 좋게 출력"""
    return "\n".join("[ " + "  ".join(f"{x:.{prec}g}" for x in row) + " ]" for row in M)


def det_bareiss(Ain):
    """Bareiss 알고리즘을 이용한 행렬식 계산"""
    A = [[Fraction(x) for x in row] for row in Ain]
    n = len(A)
    sign, denom = 1, Fraction(1)
    for k in range(n - 1):
        # 피벗 탐색
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
    """소행렬 생성"""
    return [row[:c] + row[c + 1 :] for i, row in enumerate(A) if i != r]

def adjugate_inverse(A):
    """행렬식과 수반행렬을 이용해 역행렬 계산"""
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
    """두 행렬이 같은지 (오차 허용 범위 내) 비교"""
    return all(abs(A[i][j] - B[i][j]) <= tol for i in range(len(A)) for j in range(len(A)))

def verify_inverse(A, inv):
    """A × A⁻¹ = I 검증"""
    n = len(A)
    result = [[sum(A[i][k] * inv[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    print("\n[추가기능] A × A⁻¹ 결과 (단위행렬 검증):")
    print(pretty_matrix(result))

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
        same = matrices_close(inv1, inv2)
        print("\n두 결과 동일 여부:", same)
        # 추가기능: 단위행렬 검증
        verify_inverse(A, inv2)
    else:
        print("\n비교 불가")


if __name__ == "__main__":
    main()
