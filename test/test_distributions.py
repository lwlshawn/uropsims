import generate

DIST1 = generate.uniform_bn(3)

DIST2 = generate.constructed_reference_bn(3)

DIST3 = [[0] * 3 for _ in range(2)]
DIST3[0][0] = 0.5
DIST3[1][0] = 0.5
DIST3[0][1] = 2/5
DIST3[1][1] = 1/3
DIST3[0][2] = 1/7
DIST3[1][2] = 1/6


DIST4 = [[0] * 3 for _ in range(2)]
DIST4[0][0] = 0.5
DIST4[1][0] = 0.5
DIST4[0][1] = 1/4
DIST4[1][1] = 3/4
DIST4[0][2] = 1/4
DIST4[1][2] = 3/4

DELTA = 10**(-14)
