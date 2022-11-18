import curses


def main(stdscr):
    s = curses.initscr()

    sh, sw = stdscr.getmaxyx()

    stdscr.clear()

    stdscr.addch(sh // 2, sw // 2, curses.ACS_PI)
    stdscr.addch(sh // 2, sw // 2 + 1, curses.ACS_BLOCK)

    stdscr.refresh()

    stdscr.getch()


if __name__ == "__main__":
    curses.wrapper(main)
