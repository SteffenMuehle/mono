#rotate both monitors left
xrandr -o  left

#"unrotate" left monitor
xrandr --output DP-7 --rotate normal

#left monitor left of right monitor
xrandr --output DP-7 --left-of DP-0

# set interface sizing: dpi = dots per inch
xrandr --dpi 180
