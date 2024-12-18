# source("02.setup.R")
# ![](https://github-production-user-asset-6210df.s3.amazonaws.com/1679452/396745694-77bce463-d4df-40f2-9966-6da2b1676739.jpg)
# ![](https://github.com/user-attachments/assets/77bce463-d4df-40f2-9966-6da2b1676739)

dots_overlay <- dart |>
  ggplot() +
  aes(x = longitude, y = latitude, color = winslow) +
  scale_color_manual(values = c("#FF8C42", "#36B37E"), guide = "none") +
  ggpubr::background_image(dartmouth_img) +
  geom_point(size = 0.5) +
  coord_dartmouth +
  bg_transparent()

ggsave(
  file.path(img_dir, "dots_overlay.png"), 
  dots_overlay,
  width = 8, height = 6, dpi = 300
)

dots_only <- dart |>
  ggplot() +
  aes(x = longitude, y = latitude, color = winslow) +
  scale_color_manual(values = c("#FF8C42", "#36B37E"), guide = "none") +
  geom_point(size = 0.5) +
  coord_dartmouth +
  bg_transparent()

ggsave(
  file.path(img_dir, "dots_only.png"), 
  dots_only,
  width = 8, height = 6, dpi = 300
)

segments_plot <- ggplot(dartmouth_segs) +
  geom_segment(
    color = "#f07178",
    aes(x = x, y = y, xend = xend, yend = yend),
    arrow = arrow(type = "closed", length = unit(0.1, "cm"))
  ) +
  coord_dartmouth +
  bg_transparent()

ggsave(
  file.path(img_dir, "segments_plot.png"), 
  segments_plot,
  width = 8, height = 6, dpi = 300
)

dartmouth_rose <- plot_rose(dartmouth, "", labels = c("N", "E", "S", "W"))
ggsave(
  file.path(img_dir, "dartmouth_rose.png"), 
  dartmouth_rose,
  width = 8, height = 6, dpi = 300
)

rose_nwbw <- dartmouth |>
  mutate(color = if_else((row_number()) != 28, "#004B59", "#f07178")) |> 
  plot_rose("", labels = c("N", "E", "S", "W"), highlight = TRUE) +
  geom_text(x = 300, y = 470, label = "NWbW", color = "#EBEBEB", size = 4)

ggsave(
  file.path(img_dir, "rose_nwbw.png"), 
  rose_nwbw,
  width = 8, height = 6, dpi = 300
)

segments_highlight_nwbw <- dartmouth_segs |>
  mutate(nwbw = group == 28) |>
  ggplot() +
  aes(x = x, y = y, xend = xend, yend = yend, color = nwbw) +
  geom_segment(arrow = arrow(type = "open", length = unit(0.1, "cm"))) +
  scale_color_manual(values = c("#004B59", "#f07178"), guide = "none") +
  coord_dartmouth +
  bg_transparent() +
  theme(
    panel.grid = element_blank(),
  ) 

ggsave(
  file.path(img_dir, "segments_highlight_nwbw.png"), 
  segments_highlight_nwbw,
  width = 8, height = 6, dpi = 300
)

segments_highlight_nne <- dartmouth_segs |>
  mutate(nne = group == 3) |>
  ggplot() +
  aes(x = x, y = y, xend = xend, yend = yend, color = nne) +
  geom_segment(arrow = arrow(type = "open", length = unit(0.1, "cm"))) +
  scale_color_manual(values = c("#004B59", "#f07178"), guide = "none") +
  coord_dartmouth +
  bg_transparent() +
  theme(
    panel.grid = element_blank(),
  )

ggsave(
  file.path(img_dir, "segments_highlight_nne.png"), 
  segments_highlight_nne,
  width = 8, height = 6, dpi = 300
)


rose_nne <- dartmouth |>
  mutate(color = if_else((row_number()) != 3, "#004B59", "#f07178")) |>
  plot_rose("", labels = c("N", "E", "S", "W"), highlight = TRUE) +
  geom_text(x = 22, y = 750, label = "NNE", color = "#EBEBEB", size = 4)

ggsave(
  file.path(img_dir, "rose_nne.png"), 
  rose_nne,
  width = 8, height = 6, dpi = 300
)

all_roses <- cowplot::plot_grid(
  plotlist = purrr::map2(
    bearings_ls,
    names(bearings_ls),
    plot_rose,
    size_x = 7,
    size_title = 9,
    type = "all"
  )
)

ggsave(
  file.path(img_dir, "all_roses.png"), 
  all_roses,
  width = 8, height = 6, dpi = 300
)

north <- hemi |> 
  filter(num_bins == n_groups, hemisphere == "north") |> 
  plot_rose("Northern hemisphere", labels = c("N", "E", "S", "W"))

south <- hemi |> 
  filter(num_bins == n_groups, hemisphere == "south") |> 
  plot_rose("Southern hemisphere", labels = c("N", "E", "S", "W"))

ggsave(
  file.path(img_dir, "hemisphere.png"), 
  cowplot::plot_grid(north, south, ncol = 2),
  width = 12, height = 6, dpi = 300
)
