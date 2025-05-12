import yaml
from copy import deepcopy

with open("template.yaml", "r") as f:
    config = yaml.safe_load(f)

for options in [(True, False), (False, False)]:
    new_config = deepcopy(config)
    new_config["config"]["no_parent_queries"] = options[0]
    new_config["config"]["no_sibling_queries"] = options[1]
    name = f"multihop_{'no_parent' if options[0] else 'parent'}"
    with open(f"./{name}.yaml", "w") as f:
        yaml.dump(new_config, f)