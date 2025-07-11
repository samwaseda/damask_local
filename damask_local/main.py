import difflib
import requests
import yaml
import warnings
import numpy as np
from pathlib import Path

from damask import YAML, ConfigMaterial, Rotation, GeomGrid, seeds
from mendeleev.fetch import fetch_table


def list_elasticity(
    sub_folder="elastic",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the elasticity YAML files in the specified directory from the
    specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML
            files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML
            files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """
    return get_yaml(sub_folder, repo_owner, repo_name, directory_path)


def list_plasticity(
    sub_folder="plastic",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the plasticity YAML files in the specified directory from the specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """
    return get_yaml(sub_folder, repo_owner, repo_name, directory_path)


def get_yaml(
    sub_folder="",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the YAML files in the specified directory from the specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """

    # GitHub API URL to get the directory contents
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}/{sub_folder}"

    # Dictionary to store YAML content
    yaml_dicts = {}

    # Fetch directory contents
    response = requests.get(api_url)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["name"].endswith(".yaml"):
                # Get raw file URL
                raw_url = file["download_url"]

                # Download the file
                file_response = requests.get(raw_url)
                if file_response.status_code == 200:
                    try:
                        # Load the YAML content into a Python dictionary
                        yaml_content = yaml.safe_load(file_response.text)
                        yaml_dicts[file["name"].replace(".yaml", "")] = yaml_content
                    except yaml.YAMLError as e:
                        warnings.warn(f"Failed to load {file['name']}: {e}")
                else:
                    warnings.warn(
                        f"Failed to download {file['name']}: {file_response.status_code}"
                    )
    else:
        response.raise_for_status()
    return yaml_dicts


def get_phase(
    composition, elasticity, plasticity=None, lattice=None, output_list=None
):
    """
    Returns a dictionary describing the phases for damask.

    For the details of isotropic model, one can refer to:
    https://doi.org/10.1016/j.scriptamat.2017.09.047
    """
    if lattice is None:
        lattice = {"BCC": "cI", "HEX": "hP", "FCC": "cF"}[
            get_atom_info(name=composition)["lattice_structure"]
        ]
    if output_list is None:
        if plasticity is None:
            output_list = ["F", "P", "F_e"]
        else:
            output_list = ["F", "P", "F_e", "F_p", "L_p", "O"]
    d = {
        composition: {
            "lattice": lattice,
            "mechanical": {"output": output_list, "elastic": elasticity},
        }
    }
    if plasticity is not None:
        d[composition]["mechanical"]["plastic"] = plasticity
    return d


def get_atom_info(difflib_cutoff=0.8, **kwargs):
    """
    Get atomic information from the periodic table.

    Args:
        difflib_cutoff (float): Cutoff for difflib.get_close_matches
        **kwargs: Key-value pairs to search for

    Returns:
        dict: Atomic information
    """
    df = fetch_table("elements")
    if len(kwargs) == 0:
        raise ValueError("No arguments provided")
    for key, tag in kwargs.items():
        if difflib_cutoff < 1:
            key = get_tag(key, df.keys(), cutoff=difflib_cutoff)
            tag = get_tag(tag, df[key], cutoff=difflib_cutoff)
            if sum(df[key] == tag) == 0:
                raise KeyError(f"'{tag}' not found")
            df = df[df[key] == tag]
    return df.squeeze(axis=0).to_dict()


def get_tag(tag, arr, cutoff=0.8):
    results = difflib.get_close_matches(tag, arr, cutoff=cutoff)
    if len(results) == 0:
        raise KeyError(f"'{tag}' not found")
    return results[0]


def get_rotation(method="from_random", shape=None):
    """
    Args:
        method (damask.Rotation.*/str): Method of damask.Rotation class which
            based on the given arguments creates the Rotation object. If
            string is given, it looks for the method within `damask.Rotation`
            via `getattr`.

    Returns:
        damask.Rotation: A Rotation object
    """
    if isinstance(method, str):
        method = getattr(Rotation, method)
    return method(shape=shape)


def generate_material(rotation, elements, phase, homogenization):
    _config = ConfigMaterial(
        {"material": [], "phase": phase, "homogenization": homogenization}
    )
    if not isinstance(rotation, (list, tuple, np.ndarray)):
        rotation = [rotation]
    if not isinstance(rotation, (list, tuple, np.ndarray)):
        elements = [elements]
    for r, e in zip(rotation, elements):
        _config = _config.material_add(
            O=r, phase=e, homogenization=list(homogenization.keys())[0]
        )
    return _config


def generate_load_step(
    N,
    t,
    F=None,
    dot_F=None,
    P=None,
    dot_P=None,
    f_out=None,
    r=None,
    f_restart=None,
    estimate_rate=None,
):
    """
    Args:
        N (int): Number of increments
        t (float): Time of load step in seconds, i.e.
        F (numpy.ndarray): Deformation gradient at end of load step
        dot_F (numpy.ndarray): Rate of deformation gradient during load step
        P (numpy.ndarray): First Piola–Kirchhoff stress at end of load step
        dot_P (numpy.ndarray): Rate of first Piola–Kirchhoff stress during
            load step
        r (float): Scaling factor (default 1) in geometric time step series
        f_out (int): Output frequency of results, i.e. f_out=3 writes results
            every third increment
        f_restart (int): output frequency of restart information; e.g.
            f_restart=3 writes restart information every tenth increment
        estimate_rate (float): estimate field of deformation gradient
            fluctuations based on former load step (default) or assume to be
            homogeneous, i.e. no fluctuations

    Returns:
        dict: A dictionary of the load step

    You can find more information about the parameters in the damask documentation:
    https://damask-multiphysics.org/documentation/file_formats/grid_solver.html#load-case
    """
    result = {
        "boundary_conditions": {"mechanical": {}},
        "discretization": {"t": t, "N": N},
    }
    if r is not None:
        result["discretization"]["r"] = r
    if f_out is not None:
        result["f_out"] = f_out
    if f_restart is not None:
        result["f_restart"] = f_restart
    if estimate_rate is not None:
        result["estimate_rate"] = estimate_rate
    if F is None and dot_F is None and P is None and dot_P is None:
        raise ValueError("At least one of the tensors should be provided.")
    if F is not None:
        result["boundary_conditions"]["mechanical"]["F"] = F
    if dot_F is not None:
        result["boundary_conditions"]["mechanical"]["dot_F"] = dot_F
    if P is not None:
        result["boundary_conditions"]["mechanical"]["P"] = P
    if dot_P is not None:
        result["boundary_conditions"]["mechanical"]["dot_P"] = dot_P
    return result


def generate_grid_from_voronoi_tessellation(
    spatial_discretization, num_grains, box_size
):
    if isinstance(spatial_discretization, (int, float)):
        spatial_discretization = np.array(3 * [spatial_discretization])
    if isinstance(box_size, (int, float)):
        box_size = np.array(3 * [box_size])
    seed = seeds.from_random(box_size, num_grains)
    return GeomGrid.from_Voronoi_tessellation(
        spatial_discretization, box_size, seed
    )


def get_loading(solver, load_steps):
    if not isinstance(load_steps, list):
        load_steps = [load_steps]
    return YAML(solver=solver, loadstep=load_steps)


def get_homogenization(method=None, parameters=None):
    """
    Returns damask homogenization as a dictionary.
    Args:
        method(str): homogenization method
        parameters(dict): the required parameters
    """
    if method is None:
        method = "SX"
    if parameters is None:
        parameters = {"N_constituents": 1, "mechanical": {"type": "pass"}}
    return {method: parameters}


def generate_loading_tensor(default="F"):
    """
    Returns the default boundary conditions for the damask loading tensor.

    Args:
        default (str): Default value of the tensor. It can be 'F', 'P', 'dot_F'
            or 'dot_P'.

    Returns:
        tuple: A tuple of two numpy arrays. The first array is the keys and the
            second array is the values.
    """
    assert default in ["F", "P", "dot_F", "dot_P"]
    if default == "F":
        return np.full((3, 3), "F").astype("<U5"), np.eye(3)
    else:
        return np.full((3, 3), default).astype("<U5"), np.zeros((3, 3))


def loading_tensor_to_dict(key, value):
    """
    Converts the damask loading tensor to a dictionary.

    Args:
        key (numpy.ndarray): Keys of the tensor
        value (numpy.ndarray): Values of the tensor

    Returns:
        dict: A dictionary of the tensor

    Example:
        key, value = generate_loading_tensor()
        loading_tensor_to_dict(key, value)

    Comments:

        `key` and `value` should be generated from
        `generate_loading_tensor()` and as the format below:

        (array([['F', 'F', 'F'],
                ['F', 'F', 'F'],
                ['F', 'F', 'F']], dtype='<U5'),
         array([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]]))

        where the first array is the keys and the second array is the values.
        The keys can be 'F', 'P', 'dot_F' or 'dot_P'. These keys correspond to:

        F: deformation gradient at end of load step
        dot_F: rate of deformation gradient during load step
        P: first Piola–Kirchhoff stress at end of load step
        dot_P: rate of first Piola–Kirchhoff stress during load step
    """
    result = {}
    for tag in ["F", "P", "dot_F", "dot_P"]:
        if tag in key:
            mat = np.full((3, 3), "x").astype(object)
            mat[key == tag] = value[key == tag]
            result[tag] = mat.tolist()
    return result


def get_elasticity(key="Hooke_Al"):
    return list_elasticity()[key]


def get_plasticity(key="phenopowerlaw_Al"):
    return list_plasticity()[key]



def save_material(
    rotation, composition, phase, homogenization, path, file_name="material.yaml"
):
    material = generate_material([rotation], [composition], phase, homogenization)
    material.save(path / file_name)
    return file_name


def save_grid(
    box_size, spatial_discretization, num_grains, path, file_name="damask"
):
    grid = generate_grid_from_voronoi_tessellation(
        box_size=box_size,
        spatial_discretization=spatial_discretization,
        num_grains=num_grains,
    )
    grid.save(path / file_name)
    return file_name


def save_loading(path, strain=1.0e-3, file_name="loading.yaml"):
    keys, values = generate_loading_tensor("dot_F")
    values[0, 0] = strain
    keys[1, 1] = keys[2, 2] = "P"
    data = loading_tensor_to_dict(keys, values)
    load_step = [
        generate_load_step(N=40, t=10, f_out=4, **data),
        generate_load_step(N=20, t=20, f_out=4, **data),
    ]
    loading = get_loading(solver={"mechanical": "spectral_basic"}, load_steps=load_step)
    loading.save(path / file_name)
    return file_name


def run_damask(material, loading, grid):
    command = f"DAMASK_grid -m {material} -l {loading} -g {grid}.vti".split()
    import subprocess

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=path
    )
    stdout, stderr = process.communicate()
    return process, stdout, stderr


def average(d):
    return np.average(list(d.values()), axis=1)


def get_hdf_file_name(material, loading, grid):
    return "{}_{}_{}.hdf5".format(grid, loading.split(".")[0], material.split(".")[0])


def get_results(file_name, path):
    results = Result(path / file_name)
    results.add_stress_Cauchy()
    results.add_strain()
    results.add_equivalent_Mises("sigma")
    results.add_equivalent_Mises("epsilon_V^0.0(F)")
    stress = average(results.get("sigma"))
    strain = average(results.get("epsilon_V^0.0(F)"))
    stress_von_Mises = average(results.get("sigma_vM"))
    strain_von_Mises = average(results.get("epsilon_V^0.0(F)_vM"))
    return stress, strain, stress_von_Mises, strain_von_Mises
