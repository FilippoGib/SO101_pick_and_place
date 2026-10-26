"""Task registrations exposed to the Isaac Lab CLI."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["isaaclab_tutorial.tasks.place_vial.mdp", "isaaclab_tutorial.tasks.place_cube.mdp"])

from isaaclab_tutorial.tasks.place_vial.config import so101 as _vial_so101  # noqa: F401
from isaaclab_tutorial.tasks.place_cube.config import so101 as _cube_so101  # noqa: F401
