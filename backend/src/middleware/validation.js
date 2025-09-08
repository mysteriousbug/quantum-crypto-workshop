const validateBB84Request = (req, res, next) => {
  const { nBits, distance, withEavesdropper } = req.body;
  
  // Validate nBits
  if (nBits !== undefined) {
    if (!Number.isInteger(nBits) || nBits < 10 || nBits > 1000) {
      return res.status(400).json({
        success: false,
        error: 'nBits must be an integer between 10 and 1000'
      });
    }
  }
  
  // Validate distance
  if (distance !== undefined) {
    if (!Number.isFinite(distance) || distance < 1 || distance > 500) {
      return res.status(400).json({
        success: false,
        error: 'distance must be a number between 1 and 500 km'
      });
    }
  }
  
  // Validate eavesdropper flag
  if (withEavesdropper !== undefined && typeof withEavesdropper !== 'boolean') {
    return res.status(400).json({
      success: false,
      error: 'withEavesdropper must be a boolean'
    });
  }
  
  next();
};

export default {
  validateBB84Request
};