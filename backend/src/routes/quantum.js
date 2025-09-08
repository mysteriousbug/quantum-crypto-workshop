const express = require('express');
const router = express.Router();
const QuantumService = require('../services/quantumService');
const { validateBB84Request } = require('../middleware/validation');

const quantumService = new QuantumService();

// BB84 Protocol Endpoint
router.post('/bb84', validateBB84Request, async (req, res) => {
  try {
    const { nBits = 100, withEavesdropper = false, distance = 50 } = req.body;
    
    console.log(`Running BB84: ${nBits} bits, distance: ${distance}km, eve: ${withEavesdropper}`);
    
    const result = await quantumService.runBB84Protocol({
      nBits,
      withEavesdropper,
      distance
    });
    
    res.json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('BB84 Error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// Quantum Key Generation
router.post('/generate-key', async (req, res) => {
  try {
    const { keyLength = 256 } = req.body;
    const result = await quantumService.generateQuantumKey(keyLength);
    
    res.json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Quantum Random Number Generation
router.post('/quantum-random', async (req, res) => {
  try {
    const { numBits = 8, numSamples = 100 } = req.body;
    const result = await quantumService.generateQuantumRandom(numBits, numSamples);
    
    res.json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Protocol Comparison
router.get('/protocols/compare', async (req, res) => {
  try {
    const comparison = {
      BB84: {
        yearIntroduced: 1984,
        security: 'Information-theoretic',
        keyRate: 'Medium',
        implementation: 'Most common',
        advantages: ['Well-studied', 'Widely implemented', 'Good performance'],
        disadvantages: ['Vulnerable to PNS attacks with weak coherent pulses']
      },
      E91: {
        yearIntroduced: 1991,
        security: 'Information-theoretic + Bell test',
        keyRate: 'Lower than BB84',
        implementation: 'More complex',
        advantages: ['Built-in security test', 'Entanglement-based'],
        disadvantages: ['Requires entangled photon pairs', 'Lower key rates']
      },
      SARG04: {
        yearIntroduced: 2004,
        security: 'Enhanced against PNS',
        keyRate: 'Similar to BB84',
        implementation: 'BB84 variant',
        advantages: ['Better security with weak coherent pulses', 'Compatible with BB84 hardware'],
        disadvantages: ['Slightly more complex protocol']
      }
    };
    
    res.json({ success: true, data: comparison });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;