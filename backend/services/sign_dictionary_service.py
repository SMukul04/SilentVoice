"""Service providing a read-only abstraction for the ISL Sign Dictionary."""

from typing import List, Optional

from backend.schemas.sign_dictionary import (
    SignEntry,
    SignNotFoundError,
    DuplicateSignError
)


class SignDictionaryService:
    """A data-driven dictionary abstraction representing ISL signs.
    
    Provides deterministic lookup and prevents internal state mutation.
    This strictly defines vocabulary identity, completely independent of 
    phrase resolution, model inference, or 3D avatars.
    """
    
    def __init__(self, signs: Optional[List[SignEntry]] = None):
        """Initialize the dictionary with a list of signs.
        
        Args:
            signs: A list of SignEntry objects. If None, loads a default verified vocabulary.
            
        Raises:
            DuplicateSignError: If duplicate sign_id or canonical_name entries are detected.
        """
        self._by_id: dict[str, SignEntry] = {}
        self._by_name: dict[str, SignEntry] = {}
        
        if signs is None:
            # We explicitly define a small verified vocabulary.
            # This does NOT claim complete ISL coverage.
            signs = [
                SignEntry(sign_id="HELLO", canonical_name="hello"),
                SignEntry(sign_id="THANK_YOU", canonical_name="thank you"),
                SignEntry(sign_id="PLEASE", canonical_name="please"),
                SignEntry(sign_id="YES", canonical_name="yes"),
                SignEntry(sign_id="NO", canonical_name="no")
            ]
            
        for sign in signs:
            # Validate uniqueness of sign_id
            if sign.sign_id in self._by_id:
                raise DuplicateSignError(f"Duplicate sign_id detected: {sign.sign_id}")
            
            # Validate uniqueness of canonical_name (case-insensitive)
            norm_name = sign.canonical_name.lower().strip()
            if norm_name in self._by_name:
                raise DuplicateSignError(f"Duplicate canonical_name detected: {norm_name}")
                
            self._by_id[sign.sign_id] = sign
            self._by_name[norm_name] = sign

    def get_by_id(self, sign_id: str) -> SignEntry:
        """Retrieve a sign by its exact sign_id.
        
        Args:
            sign_id: The exact case-sensitive sign identifier.
            
        Returns:
            A safe copy of the SignEntry.
            
        Raises:
            SignNotFoundError: If the sign_id is not in the dictionary.
        """
        sign = self._by_id.get(sign_id)
        if not sign:
            raise SignNotFoundError(f"Sign ID not found: {sign_id}")
        return sign.model_copy()

    def get_by_name(self, canonical_name: str) -> SignEntry:
        """Retrieve a sign by its canonical name (case-insensitive).
        
        Args:
            canonical_name: The name of the sign.
            
        Returns:
            A safe copy of the SignEntry.
            
        Raises:
            SignNotFoundError: If the canonical_name is not in the dictionary.
        """
        norm_name = canonical_name.lower().strip()
        sign = self._by_name.get(norm_name)
        if not sign:
            raise SignNotFoundError(f"Sign name not found: '{canonical_name}'")
        return sign.model_copy()

    def contains(self, identifier: str) -> bool:
        """Check if a sign exists by ID or case-insensitive name.
        
        Args:
            identifier: The sign_id or canonical_name to check.
            
        Returns:
            True if the sign exists, False otherwise.
        """
        if identifier in self._by_id:
            return True
            
        norm_name = identifier.lower().strip()
        if norm_name in self._by_name:
            return True
            
        return False

    def all_signs(self) -> List[SignEntry]:
        """Return a safe copy of all signs in the dictionary.
        
        Returns:
            A list containing copies of all SignEntry objects.
        """
        return [sign.model_copy() for sign in self._by_id.values()]
