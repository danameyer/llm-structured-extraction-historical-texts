from typing import Optional, Dict, List, Union
JsonSchemaType = Union[str, List[str]]


class Message:
    def __init__(self, role: str,
                 content: str,
                 name: str = None,
                 function_call_id: str = None,
                 function_call_arguments: str = None):
        self.role: str = role
        self.content: str = content
        self.name: str = name
        self.function_call_id: str = function_call_id
        self.function_call_arguments: str = function_call_arguments


class SimpleChatGptMessage:
    def __init__(self, role: str, content: str, name: str = None):
        self.role: str = role
        self.content: str = content
        self.name: str = name


class FunctionBuilder:
    def __init__(self,
                 tool_type: str,
                 strict: bool = None,
                 function: Optional["Function"] = None):
        self.type: str = tool_type
        if function:
            self.function: "Function" = function
        else:
            self.function: Optional["Function"] = None
        self.strict: bool = strict


class Function:
    def __init__(self,
                 name: str,
                 description: str,
                 parameters: Optional["Parameter"] = None):
        self.name: str = name
        self.description: str = description
        if parameters:
            self.parameters: "Parameter" = parameters
        else:
            self.parameters: Optional["Parameter"] = None


class Property:
    def __init__(
        self,
        property_type: JsonSchemaType,
        description: str,
        items: Optional["Property"] = None,
        properties: Optional[Dict[str, "Property"]] = None,
        default_value=None,
        enum: Optional[List[str]] = None,
        required: Optional[List[str]] = None,
        additional_properties: Optional[bool] = None,
    ):
        self.type = property_type
        self.description = description
        self.items = items
        self.properties = properties or {}
        self.default_value = default_value
        self.enum = enum
        self.required = required
        self.additional_properties = additional_properties

    def to_schema(self) -> Dict:
        schema = {"type": self.type, "description": self.description}

        if self.enum is not None:
            schema["enum"] = self.enum

        if self.items is not None:
            schema["items"] = self.items.to_schema()

        if self.properties:
            schema["properties"] = {name: prop.to_schema() for name, prop in self.properties.items()}

        if self.required is not None:
            schema["required"] = self.required

        if self.additional_properties is not None:
            schema["additionalProperties"] = self.additional_properties

        return schema


class Parameter:
    def __init__(
        self,
        parameter_type: JsonSchemaType,
        additionalProperties: bool,
        properties: Optional[Dict[str, Property]] = None,
        required: Optional[List[str]] = None,
    ):
        self.type = parameter_type
        self.properties = properties or {}
        self.required = required or []
        self.additionalProperties = additionalProperties

    def to_schema(self) -> Dict:
        return {
            "type": self.type,
            "properties": {name: prop.to_schema() for name, prop in self.properties.items()},
            "required": self.required,
            "additionalProperties": self.additionalProperties,
        }

    def add_property(
        self,
        name: str,
        property_type: str,
        description: str,
        required: bool,
    ):
        if required:
            self.required.append(name)

        self.properties[name] = Property(property_type=property_type, description=description)
