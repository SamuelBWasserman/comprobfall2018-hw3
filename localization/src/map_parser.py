def parse_map(map_file):
    with open(map_file, 'r') as map:
        lines = map.readlines()
        obstacles = []
        num_obstacles = 0
        # assign corners
        corners = lines[0].split()

        lines = iter(lines)
        # Skip first two lines
        next(lines)
        next(lines)

        for line in lines:
            num_obstacles = num_obstacles + 1
            obstacles.append(line.split())
        return corners, obstacles, num_obstacles
