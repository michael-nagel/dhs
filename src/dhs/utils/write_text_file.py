# !/usr/bin/env python3
# -*- coding: utf-8 -*-


def write_text_file(
    file: str,
    body: str,
    first_line: None | int = 1,
    last_line: None | int = -1,
) -> None:
    """
    Write text to a text file.

    This function can be particulary useful to store TeX table code in
    a text file. For example, the first and the last line can be
    stripped to embed the file within a TeX tabular environment.

    Parameters
    ----------
    file : str
        Name (path) of file.
    body : str
        Text to write.
    first_line : int | None, default 1
        First line to write.
    last_line : int | None, default -1
        Last line to write.

    Returns
    -------
    None

    See Also
    --------
    pandas.DataFrame.to_latex : Render object to a LaTeX tabular,
        longtable, or nested table.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1.5, 0.1234], "B": [-2, 15000]})
    >>> df.head()
    ...
            A      B
    0  1.5000     -2
    1  0.1234  15000
    >>> write_text_file(
            file="df.tex",
            body=df.to_latex(),
            first_line=1,
            last_line=-1
        )
    """
    with open(file, "w") as tf:
        tf.write(body)

    with open(file, "r") as tf:
        content = tf.read().splitlines(True)

    with open(file, "w") as tf:
        tf.writelines(content[first_line:last_line])
        tf.close()
