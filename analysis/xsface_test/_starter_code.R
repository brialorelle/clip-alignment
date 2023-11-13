library(tidyverse)
library(here)
library(glue)

naming_files <- list.files(here::here("data/xs-face/naming"), pattern = "*.csv") |>
  str_replace(".csv", "")

GET_FRAMES = TRUE

for (naming_file in naming_files[1]) {
  if (GET_FRAMES) folder <- "frames" else "clips"
  if (dir.exists(here(folder, naming_file))) next

  dir.create(here(folder, naming_file), recursive = TRUE)

  naming_events <- read_csv(here("data/xs-face/naming", glue("{naming_file}.csv")),
                            col_types = "cc")
  naming_events <- naming_events |>
    separate(time,
             if(str_count(naming_events$time[1], ":") == 2) {
               c(NA, "min", "s")
             } else c("min", "s"),
             sep = ":") |>
    mutate(time = as.numeric(min) * 60 + as.numeric(s))

  if (GET_FRAMES) {
    select_text = lapply(naming_events$time |> unique(), \(x) {
      times = c(x-2, x-1, x, x+1, x+2)
      lapply(times, \(y) {
        glue("lt(prev_pts*TB\\,{y})*gte(pts*TB\\,{y})")
      }) |> paste(collapse = "+")
    }) |> paste(collapse = "+")

    system(glue("ffmpeg -i videos/{naming_file}_objs.mov -an -vf ",
                "select='{select_text}' -fps_mode passthrough ",
                "-frame_pts true frames/{naming_file}/frame_%d.jpg"))
  } else {
    lapply(naming_events$time |> unique(), \(x) {
      system(glue("ffmpeg -ss {x-2} -i videos/{naming_file}_objs.mov ",
                  "-fps_mode passthrough -t 4 clips/{naming_file}/clip_{x}.mov"))
    })
  }

}

all_naming_events <- lapply(naming_files, \(naming_file) {
  naming_events <- read_csv(here("naming", glue("{naming_file}.csv")),
                            col_types = "cc") |>
    select(where(\(x) {all(!is.na(x))}))
  naming_events |>
    separate(time,
             if(str_count(naming_events$time[1], ":") == 2) {
               c(NA, "min", "s")
             } else c("min", "s"),
             sep = ":") |>
    mutate(time = as.numeric(min) * 60 + as.numeric(s),
           child_id = naming_file)
}) |> bind_rows() |> select(child_id, everything())

all_naming_events <- all_naming_events |>
  separate(name, c("name", NA), sep = " ", extra = "drop", fill = "right") |>
  mutate(name = sub("tooth", "", name))

write_csv(all_naming_events, "all_naming_events.csv")

objs <- c("ball", "brush", "car", "cat",
          "gimo", "manu", "tima", "zem")

# for (naming_file in naming_files) {
#   for (obj in objs) {
#     dir.create(here("frames", naming_file, obj), recursive = TRUE)
#   }
# }

all_naming_frames <- list()

for (naming_file in naming_files) {
  for (obj in objs) {
    frames <- list.files(here("frames", naming_file, obj),
                         pattern = "*jpg") |>
      str_replace(".jpg", "")
    df <- tibble(child_id = naming_file,
                 name = obj,
                 time = frames)
    all_naming_frames <- c(all_naming_frames, list(df))
  }
}
all_naming_frames <- bind_rows(all_naming_frames)



e <- all_naming_events |>
  left_join(all_naming_frames |>
              mutate(time = as.double(time)),
            by = c("child_id", "time"))
pivot_wider(e, names_from = "name.y",
            values_from = "name.y", values_fn = length) |>
  write_csv("all_namings.csv")



all_namings <- read_csv("all_namings.csv")
all_namings_cleaned <- all_namings |>
  fill(utterance, .direction = "down") |>
  mutate(n_objs = rowSums(across(zem:car), na.rm = TRUE)) |>
  filter(n_objs >= 1) |>
  pivot_longer(cols = zem:car,
               names_to = "img_label") |>
  filter(!is.na(value)) |>
  select(child_id, time, utterance, txt_label = name, img_label, n_objs) |>
  filter(!(txt_label %in% c("cat", "gimo")),
         !(img_label %in% c("cat", "gimo")))
all_namings_cleaned |> write_csv("all_namings_cleaned.csv")
