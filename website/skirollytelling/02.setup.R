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
dart_url <- "https://github.com/user-attachments/assets/636f8cd2-31a6-4bd8-8a9f-1cd662d6a295"
download.file(dart_url, destfile = "../images/dartmouth.png")
dartmouth_img <- png::readPNG("../images/dartmouth.png", native = TRUE)
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
