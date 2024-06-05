from typing import Optional, Dict, List


class Message:
    def __init__(self, role: str, content: str, name: str = None):
        self.role: str = role
        self.content: str = content
        self.name: str = name


class FunctionBuilder:
    def __init__(self, name: str, description: str, parameters: Optional["Parameter"] = None):
        self.name: str = name
        self.description: str = description
        if parameters:
            self.parameters: "Parameter" = parameters
        else:
            self.parameters: Optional["Parameter"] = None


class Property:
    def __init__(self,
                 property_type: str,
                 description: str,
                 items: 'Property' = None,
                 properties: dict[str, 'Property'] = None):
        self.type: str = property_type
        self.description: str = description
        if items:
            self.items: 'Property' = items
        else:
            self.items: Optional["Property"] = None
        if properties:
            self.properties: dict[str, 'Property'] = properties
        else:
            self.properties: Dict[str, Property] = dict()


class Parameter:
    def __init__(self, parameter_type: str,
                 properties: Optional[Dict[str, Property]] = None,
                 required: Optional[List[str]] = None):

        self.type: str = parameter_type

        if properties:
            self.properties: Dict[str, Property] = properties
        else:
            self.properties: Dict[str, Property] = dict()

        if required:
            self.required: List[str] = required
        else:
            self.required: List[str] = list()

    def add_property(self, name: str, property_type: str, description: str, required: List[str]):
        if required:
            self.required.append(name)

        self.properties[name] = Property(property_type, description)