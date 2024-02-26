#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Imports

import os
import unittest


class EndToEndTest(unittest.TestCase):
    def test_num_of_gen_figs(self) -> None:
        """
        Perform end-to-end test #1.

        This function compares the number of figures generated to the
        asserted number.
        """
        figs = [
            fig
            for fig in os.listdir("reports/figures")
            if fig.startswith("fig")
        ]
        self.assertEqual(len(figs), 5)

    def test_num_of_gen_tbls(self) -> None:
        """
        Perform end-to-end test #2.

        This function compares the number of tables generated to the
        asserted number.
        """
        tabs = [
            tab
            for tab in os.listdir("reports/tables")
            if tab.startswith("tab")
        ]
        self.assertEqual(len(tabs), 4)


if __name__ == "__main__":
    unittest.main()
