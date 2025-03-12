from collections import Counter

def solution(points, routes):
    robot_paths = []
    for route in routes:
        time = 0

        # 이동할 로봇 위치 및 포지션
        for i in range(len(route) -1):
            start_x, start_y = points[route[i] - 1]
            end_x, end_y = points[route[i + 1] - 1]

            while start_x != end_x:
                robot_paths.append((start_x, start_y, time))
                if start_x < end_x:
                    start_x += 1
                else:
                    start_x -= 1
                time += 1

            while start_y != end_y:
                robot_paths.append((start_x, start_y, time))
                if start_y < end_y:
                    start_y += 1
                else:
                    start_y -= 1
                time += 1

        robot_paths.append((start_x, start_y, time))
    res = 0
    temp = Counter(robot_paths)
    for i in temp.values():
        if i >= 2:
            res += 1
    return res

if __name__ == "__main__":
    print(solution([[3, 2], [6, 4], [4, 7], [1, 4]], [[4, 2], [1, 3], [4, 2], [4, 3]]))
