import ast

def parse_map(map_file):
    with open(map_file, 'r') as map:
        lines = map.readlines()
        obstacles = []
        num_obstacles = 0
        # assign corners
        corner_strings = lines[0].split()
        corners = []
        for item in corner_strings:
            corners.append(ast.literal_eval(item))

        lines = iter(lines)
        # Skip first two lines
        next(lines)
        next(lines)

        for line in lines:
            num_obstacles = num_obstacles + 1
            obstacle = []
            # print line.split(',')
            for item in line.split():
                obstacle.append(ast.literal_eval(item))
            obstacles.append(obstacle)

        return corners, obstacles, num_obstacles
