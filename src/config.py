"""
src/config.py
Projektövergripande konfiguration för maskininlärningspipelinen.

Innehåller i dagsläget central konfiguration för loggning.
Modulen är avsedd att byggas ut med projektgemensamma konstanter, 
såsom mappsökvägar, modellparametrar och datainställningar.
"""

import logging
import sys

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Konfigurerar och returnerar en standardiserad logger för projektet.
    
    Säkerställer enhetligt format och tidsstämplar över samtliga moduler
    samt förhindrar dubblerade handlers vid upprepade anrop.
    """
    logger = logging.getLogger(name)

    # Lägg bara till handler om den inte redan finns, annars dubbleras loggraderna vid varje anrop
    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger