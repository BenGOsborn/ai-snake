import curses


def main():
    s = curses.initscr()
    curses.curs_set(0)

    sh, sw = s.getmaxyx()
    window = curses.newwin(sh, sw, 0, 0)


if __name__ == "__main__":
    main()
