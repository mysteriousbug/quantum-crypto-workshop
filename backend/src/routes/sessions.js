const express = require('express');
const router = express.Router();
const { v4: uuidv4 } = require('uuid');
const AWS = require('aws-sdk');

const dynamodb = new AWS.DynamoDB.DocumentClient({
  region: process.env.AWS_REGION || 'ap-south-1'
});

const SESSIONS_TABLE = process.env.DYNAMODB_SESSIONS_TABLE || 'quantum-crypto-sessions';

// Create new session
router.post('/create', async (req, res) => {
  try {
    const sessionId = uuidv4();
    const { userId, workshopType = 'bb84-demo' } = req.body;
    
    const session = {
      sessionId,
      userId: userId || 'anonymous',
      workshopType,
      createdAt: new Date().toISOString(),
      status: 'active',
      results: [],
      metadata: {
        userAgent: req.headers['user-agent'],
        ip: req.ip
      }
    };
    
    if (process.env.NODE_ENV === 'production') {
      await dynamodb.put({
        TableName: SESSIONS_TABLE,
        Item: session
      }).promise();
    }
    
    res.json({
      success: true,
      sessionId,
      message: 'Session created successfully'
    });
  } catch (error) {
    console.error('Session creation error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Save session results
router.post('/:sessionId/results', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { results, stepCompleted } = req.body;
    
    const updateParams = {
      TableName: SESSIONS_TABLE,
      Key: { sessionId },
      UpdateExpression: 'SET #results = list_append(if_not_exists(#results, :empty_list), :new_result), #lastUpdated = :timestamp',
      ExpressionAttributeNames: {
        '#results': 'results',
        '#lastUpdated': 'lastUpdated'
      },
      ExpressionAttributeValues: {
        ':empty_list': [],
        ':new_result': [{
          step: stepCompleted,
          results,
          timestamp: new Date().toISOString()
        }],
        ':timestamp': new Date().toISOString()
      }
    };
    
    if (process.env.NODE_ENV === 'production') {
      await dynamodb.update(updateParams).promise();
    }
    
    res.json({
      success: true,
      message: 'Results saved successfully'
    });
  } catch (error) {
    console.error('Save results error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get session data
router.get('/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    if (process.env.NODE_ENV === 'production') {
      const result = await dynamodb.get({
        TableName: SESSIONS_TABLE,
        Key: { sessionId }
      }).promise();
      
      if (!result.Item) {
        return res.status(404).json({ success: false, error: 'Session not found' });
      }
      
      res.json({ success: true, session: result.Item });
    } else {
      res.json({ 
        success: true, 
        session: { 
          sessionId, 
          status: 'active',
          results: []
        } 
      });
    }
  } catch (error) {
    console.error('Get session error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;