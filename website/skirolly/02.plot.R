library(arrow)
library(dplyr)
library(ggplot2)

x1 <- -72.095
y1 <- 43.7775
x2 <- -72.101
y2 <- 43.7900
m <- (y2 - y1) / (x2 - x1)
b <- y1 - m * x1

data_dir <- "../images/data"
img_dir <- "../images"
dart_url <- "https://github.com/user-attachments/assets/636f8cd2-31a6-4bd8-8a9f-1cd662d6a295"
download.file(dart_url, destfile = file.path(img_dir, "dartmouth.png"))
dartmouth_img <- png::readPNG(file.path(img_dir, "dartmouth.png"), native = TRUE)
bearings_ls <- readRDS(file.path(data_dir, "bearings_48_ls.rds"))
dartmouth_segs <- read_parquet(file.path(data_dir, "dartmouth_segs.parquet"))
hemi <- read_parquet(file.path(data_dir, "hemisphere_roses.parquet")) |>
  tidyr::unnest(bearings) 

dart <- read_parquet(file.path(data_dir, "dartmouth_runs.parquet")) |>
  group_by(run_id) |>
  mutate(winslow = (m * longitude) + b < latitude) |>
  arrange(index)

n_groups <- 32 # number of spokes
x_range <- c(-72.1128, -72.081)
y_range <- c(43.7781, 43.7902)
length_x <- diff(x_range) # Length in x-direction
length_y <- diff(y_range) # Length in y-direction

desired_yx_ratio <- dim(dartmouth_img)[1] / dim(dartmouth_img)[2]
ratio <- (length_x / length_y) * desired_yx_ratio
coord_dartmouth <- coord_fixed(
  xlim = x_range,
  ylim = y_range,
  ratio = ratio
)

colors <- c("#f07178", "#004B59", "#FFC857", "#36B37E", "#FF8C42", "#F4F1E9", "#8A9393", "#2A2D34")

bg_transparent <- function() {
  theme_minimal() +
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      axis.title = element_blank(),
      axis.text = element_blank(),
      legend.position = "none",
    )
}
plot_rose <- function(dat, ski_area_name, size_title = 24, size_x = 20, highlight = FALSE, labels = NULL, type = NULL, hemi = NULL) {
  p <- dat |>
    ggplot() +
    aes(x = bin_center, y = bin_count) +
    coord_radial(start = -pi / 32, expand = FALSE) +
    # coord_polar(start = -pi / 32) +
    scale_x_continuous(
      breaks = seq(0, 270, 90),
      labels = labels
    ) +
    scale_y_sqrt( # scaled by area
      breaks = max(dat$bin_count)
    ) +
    labs(title = ski_area_name) +
    bg_transparent() +
    theme(
      axis.text.x = element_text(size = size_x, color = "#EBEBEB"),
      plot.title = element_text(hjust = 0.5, size = size_title, color = "#EBEBEB")
    )
  if (!is.null(type)) {
    p <- p + geom_col(fill = "#f07178") +
      theme(
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
      )
  } else if (!is.null(hemi)){
    p <- p +
      geom_col(color = "#EBEBEB", aes(fill = color)) +
      scale_fill_manual(values = hemi)
  } else if (!highlight) {
    p <- p + geom_col(color = "#EBEBEB", fill = "#f07178")
  } else {
    p <- p +
      geom_col(color = "#EBEBEB", aes(fill = color)) +
      scale_fill_identity()
  }
  p
}
dartmouth <- bearings_ls[["Dartmouth Skiway"]]
# whaleback <- bearings_ls[["Whaleback Mountain"]]
# killington <- bearings_ls[["Killington Resort"]]


## Plots ----

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
