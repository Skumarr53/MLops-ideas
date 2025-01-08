
def display_config_section(config: Any, indent: int = 0, section: str = "") -> None:
    """Recursively display configuration sections with proper indentation"""
    if isinstance(config, (dict, OmegaConf, DictConfig)):
        if section:
            print("\n" + " " * indent + f"{section}:")
        for key, value in config.items():
            display_config_section(value, indent + 2, key)
    else:
        print(" " * indent + f"{section}: {config}")

def validate_config(config: OmegaConf) -> bool:
    """Generic config validator with common checks"""
    def validate_section(section: Any, path: str = "") -> None:
        if isinstance(section, (dict, OmegaConf)):
            for key, value in section.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check for None/null values in required fields
                if value is None:
                    raise ValueError(f"Required field '{current_path}' cannot be null")
                
                # Validate date pairs if they exist
                if 'start_date' in section and 'end_date' in section:
                    if section['start_date'] > section['end_date']:
                        raise ValueError(
                            f"Start date must be before end date in section '{path}'"
                        )
                
                # Custom validation for table names
                if isinstance(value, str) and 'table' in key.lower():
                    if not value.startswith('EDS_PROD'):
                        raise ValueError(
                            f"Table '{current_path}' must start with 'EDS_PROD'"
                        )
                
                validate_section(value, current_path)
    
    try:
        validate_section(config)
        return True
    except Exception as e:
        print(f"Configuration validation failed: {str(e)}")
        return False

def user_config_check(config):
    # Load config
    
    # Display configuration
    print("\n=== PIPELINE CONFIGURATION REVIEW ===")
    display_config_section(config)
    
    print("\n=== IMPORTANT ===")
    print("Please review the configuration above carefully.")
    print("Proceeding with incorrect configuration could impact production data.")
    
    # Validate configuration
    if not validate_config(config):
        print("Exiting due to validation failure.")
        sys.exit(1)
    
    # Ask for confirmation
    while True:
        response = input("\nDo you want to proceed with this configuration? (yes/no): ").lower()
        if response in ['yes', 'no']:
            break
        print("Please enter 'yes' or 'no'")
    
    if response != 'yes':
        print("Pipeline execution cancelled by user.")
        sys.exit(0)    
    return True