

input_str = input().strip()
input_str = input_str[2:-2]
grid = []
for row in input_str.split('],['):
    grid1 = list(row.split(','))
    grid.append(grid1) 

print(grid)

