from ..utils import rgb, rgba

# cpu_color_rgb = (255, 5, 5)
# mem_color_rgb = (5, 180, 255)
#
# cpu_insert_color_rgb = (184, 0, 0)
# mem_insert_color_rgb = (0, 129, 184)
#
# cpu_delete_color_rgb = (255, 82, 82)
# mem_delete_color_rgb = (82, 203, 255)


cpu_color_rgb = (242, 189, 187)
mem_color_rgb = (101, 201, 240)

cpu_insert_color_rgb = (240, 161, 158)
mem_insert_color_rgb = (31, 155, 207)

cpu_delete_color_rgb = (247, 221, 220)
mem_delete_color_rgb = (164, 215, 237)


score_color_rgb = (250, 230, 200)

cpu_color = rgba(*cpu_color_rgb, .8)
mem_color = rgba(*mem_color_rgb, .8)

cpu_insert_color = rgba(*cpu_insert_color_rgb, .8)
mem_insert_color = rgba(*mem_insert_color_rgb, .8)

cpu_delete_color = rgba(*cpu_delete_color_rgb, .8)
mem_delete_color = rgba(*mem_delete_color_rgb, .8)

cpu_delta_color = (cpu_delete_color, cpu_insert_color)
mem_delta_color = (mem_delete_color, mem_insert_color)

socre_color = rgba(*score_color_rgb, .8)

heatmap_scale = [[0, rgb(235, 237, 240)], [0.5, rgb(64, 196, 99)], [1, rgb(33, 110, 57)]]
