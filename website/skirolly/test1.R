
# library(dplyr)
# library(ggplot2)
# data_dir <- "../images/data"
# img_dir <- "../images"
# dart_url <- "https://github.com/user-attachments/assets/1a02ca26-7034-4d87-bc0c-01f6bed997f7"
# # download.file(dart_url, destfile = file.path(img_dir, "dartmouth.png"))
# # dartmouth_img <- png::readPNG(file.path(img_dir, "dartmouth.png"), native = TRUE)
# 
# x1 <- -72.095
# y1 <- 43.7775
# x2 <- -72.101
# y2 <- 43.7900
# m <- (y2 - y1) / (x2 - x1)
# b <- y1 - m * x1
# 
# dart <- arrow::read_parquet(file.path(data_dir, "dartmouth_runs.parquet")) |>
#   group_by(run_id) |>
#   mutate(winslow = (m * longitude) + b < latitude) |>
#   arrange(index)
# 
# bg_transparent <- function() {
#   theme_minimal() +
#     theme(
#       panel.background = element_rect(fill = "transparent", colour = NA),
#       plot.background = element_rect(fill = "transparent", colour = NA),
#       axis.title = element_blank(),
#       axis.text = element_blank(),
#       legend.position = "none",
#     )
# }
# plot_rose <- function(dat, ski_area_name, size_title = 24, size_x = 20, highlight = FALSE, labels = NULL, type = NULL, hemi = NULL) {
#   p <- dat |>
#     ggplot() +
#     aes(x = bin_center, y = bin_count) +
#     coord_radial(start = -pi / 32, expand = FALSE) +
#     # coord_polar(start = -pi / 32) +
#     scale_x_continuous(
#       breaks = seq(0, 270, 90),
#       labels = labels
#     ) +
#     scale_y_sqrt( # scaled by area
#       breaks = max(dat$bin_count)
#     ) +
#     labs(title = ski_area_name) +
#     bg_transparent() +
#     theme(
#       panel.grid = element_line(color = "grey60"),
#       axis.text.x = element_text(size = size_x, color = "#EBEBEB"),
#       plot.title = element_text(hjust = 0.5, size = size_title, color = "#EBEBEB")
#     )
#   if (!is.null(type)) {
#     p <- p + geom_col(fill = "#f07178") +
#       theme(
#         panel.grid.major.x = element_blank(),
#         panel.grid.minor.x = element_blank(),
#       )
#   } else if (!is.null(hemi)){
#     p <- p +
#       geom_col(color = "#EBEBEB", aes(fill = color)) +
#       scale_fill_manual(values = hemi)
#   } else if (!highlight) {
#     p <- p + geom_col(color = "#EBEBEB", fill = "#f07178")
#   } else {
#     p <- p +
#       geom_col(color = "#EBEBEB", aes(fill = color)) +
#       scale_fill_identity()
#   }
#   p
# }

n_groups <- 32 # number of spokes
x_range <- c(-72.1065, -72.08635)
y_range <- c(43.77828, 43.7899)
length_x <- diff(x_range) # Length in x-direction
length_y <- diff(y_range) # Length in y-direction

## Plots ----
fig_height <- 3.75
fig_width <- fig_height*703/503#865/452

# desired_yx_ratio <- dim(dartmouth_img)[1] / dim(dartmouth_img)[2]
# desired_yx_ratio <- 1399/2356
# desired_yx_ratio <- 1429 / 2768
desired_yx_ratio <- 1420/1897
ratio <- (length_x / length_y) * desired_yx_ratio
coord_dartmouth <- coord_fixed(
  xlim = x_range,
  ylim = y_range,
  ratio = ratio
)

dots_only <- dart |>
  ggplot() +
  aes(x = longitude, y = latitude, color = winslow) +
  scale_color_manual(values = c("#FF8C42", "#36B37E"), guide = "none") +
  geom_point(size = 0.5) +
  coord_dartmouth +
  bg_transparent() +
  theme(
    panel.border = element_blank(),
  )

dots_overlay <- ggimage::ggbackground(
  # dots_only + theme(panel.grid = element_blank()),
  dots_only,
  dart_url
)
dots_overlay

library(grid)
g <- ggplotGrob(dots_overlay)
bg <- g$grobs[[1]]
round_bg <- roundrectGrob(x=bg$x, y=bg$y, width=bg$width, height=bg$height,
                          r=unit(0.1, "snpc"),
                          just=bg$just, name=bg$name, gp=bg$gp, vp=bg$vp)
g$grobs[[1]] <- round_bg
