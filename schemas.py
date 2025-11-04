RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "isReference": {
                "type": "BOOLEAN",
                "description": "A boolean indicating whether the text is mainly references to a scientific paper (e.g. "
                "HO, Thien; TZANETAKIS, Ioannis E. Development of a virus detection and discovery pipeline using next "
                "generation sequencing. Virology, v. 471, p. 54-60, 2014.).",
                "nullable": False,
            },
            "summary": {
                "type": "STRING",
                "nullable": False,
            },
        },
        "required": ["summary", "isReference"],
    },
}
