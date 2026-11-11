# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""``python -m ket``: the same as the ``ket`` command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
