from collections import deque
from pprint import pprint


def solution(board):
    rows, cols = len(board), len(board[0])
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    # 보드에서 시작점(R), 목표점(G) 찾기
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'R':
                start = (i,j)
            if board[i][j] == 'G':
                goal = (i,j)

    visited = [[False]*cols for _ in range(rows)]
    queue = deque([(start[0], start[1], 0)])
    visited[start[0]][start[1]] = True

    def slide(x, y, dx, dy):
        while 0 <= x+dx < rows and 0 <= y+dy < cols and board[x+dx][y+dy] != 'D':
            x += dx
            y += dy
        return x, y

    while queue:
        x, y, moves = queue.popleft()
        print("x, y, moves: " , x, y, moves)
        if (x, y) == goal:
            return moves

        for dx, dy in directions:
            print("dx, dy : " , dx, dy)
            nx, ny = slide(x, y, dx, dy)
            pprint(visited)
            if not visited[nx][ny]:
                print("moved to: ", nx, ny)
                visited[nx][ny] = True
                queue.append((nx, ny, moves+1))
            else:
                print(f"({nx}, {ny}) is already visited")

    return -1


if __name__ == "__main__":
    inp = ["...D..R", ".D.G...", "....D.D", "D....D.", "..D...."]

    assert solution(inp) == 7