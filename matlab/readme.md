# how to run this program

```matlab
% open matlab
matlab -nodesktop

% change to source file directory
cd path/to/source/file

% run main program
main
```

# describe

```
.
├── compute.m
├── data
│   └── marked.csv
├── draw.m
├── importfile.m
├── init.m
├── main.m
├── output
│   ├── fig1.fig
│   ├── fig1.svg
│   ├── fig2.fig
│   ├── fig2.svg
│   ├── fig3.fig
│   ├── fig3.svg
│   ├── fig4.fig
│   ├── fig4.svg
│   ├── fig5.fig
│   └── fig5.svg
└── readme.md
```

# system require

matlab 2017b


# emotion

```
average_label * [ favor, retweet, reply ]

nr_favor <--> user

emotion_value = sum(retweet * label)/length(label)

```
