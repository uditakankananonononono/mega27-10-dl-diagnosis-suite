\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}             % Times-compatible math
% Real Times New Roman text: Microsoft Core Fonts TTFs (EULA-permitted use),
% converted with ttf2tfm to T1/ec metrics, embedded by pdflatex.
\pdfmapfile{+tnr-ec.map}
\DeclareFontFamily{T1}{tnr}{}
\DeclareFontShape{T1}{tnr}{m}{n}{<-> tnr-ec}{}
\DeclareFontShape{T1}{tnr}{m}{it}{<-> tnri-ec}{}
\DeclareFontShape{T1}{tnr}{m}{sl}{<-> tnri-ec}{}
\DeclareFontShape{T1}{tnr}{b}{n}{<-> tnrbd-ec}{}
\DeclareFontShape{T1}{tnr}{b}{it}{<-> tnrbi-ec}{}
\DeclareFontShape{T1}{tnr}{bx}{n}{<-> tnrbd-ec}{}
\DeclareFontShape{T1}{tnr}{bx}{it}{<-> tnrbi-ec}{}
\renewcommand{\rmdefault}{tnr}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,amsmath,amsthm,amsfonts,longtable,array}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newcommand{\R}{\mathbb{R}}
\title{\textbf{Deep-Learning Diagnosis Across Diseases:\\
A CNN--GNN Hybrid Benchmark and a Confident-Learning\\Label-Noise Census of Two Public Medical Image Datasets}\\[6pt]
\large MEGA-PROGRAM-27, Item 10 (tools 4--5): malaria parasite detection and pneumonia detection}
\author{MEGA27-10b build lane \quad (computational study; no wet lab)}
\date{September 24, 2026}
