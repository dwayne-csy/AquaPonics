// sensor_server.js - Run on Windows PC (Node.js)
// Reads ESP32 via USB COM port and provides sensor data API

const express = require('express');
const cors = require('cors');
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

const app = express();
app.use(cors());
app.use(express.json());

// ==================== CONFIGURATION ====================
const CONFIG = {
    comPort: 'COM3',          // ESP32 COM port on Windows
    baudRate: 115200,
    port: 5001                // Sensor API port
};

// ==================== SENSOR DATA STORE ====================
const sensorData = {
    ph: 0.0,
    tds: 0,
    light: 0.0,
    mq135: 0,
    temperature: 0.0,
    ph_status: 'NO DATA',
    water_level: 'UNKNOWN',
    float_state: 'UNKNOWN',
    relay_states: {
        float_relay: 'OFF',
        mq135_relay: 'OFF',
        tds_relay: 'OFF',
        ph_relay: 'OFF'
    },
    sensor_connected: false,
    data_count: 0,
    last_update: 0,
    raw_data: []
};

let isConnected = false;
let serialPort = null;

// ==================== SERIAL PORT SETUP ====================
function setupSerialPort() {
    console.log(`🔌 Connecting to ${CONFIG.comPort}...`);
    
    try {
        serialPort = new SerialPort({
            path: CONFIG.comPort,
            baudRate: CONFIG.baudRate,
            autoOpen: false
        });
        
        const parser = serialPort.pipe(new ReadlineParser({ delimiter: '\n' }));
        
        serialPort.on('open', () => {
            console.log(`✅ Connected to ESP32 on ${CONFIG.comPort}`);
            isConnected = true;
        });
        
        serialPort.on('close', () => {
            console.log(`⚠️ Serial port closed`);
            isConnected = false;
        });
        
        serialPort.on('error', (err) => {
            console.error(`❌ Serial error: ${err.message}`);
            isConnected = false;
        });
        
        parser.on('data', (line) => {
            parseSensorData(line.trim());
        });
        
        serialPort.open((err) => {
            if (err) {
                console.error(`❌ Failed to open ${CONFIG.comPort}: ${err.message}`);
                if (err.message.includes('Access is denied')) {
                    console.log('   ⚠️  Close Arduino IDE Serial Monitor first!');
                }
                console.log('   💡 Check COM port number in Device Manager');
            }
        });
        
        return true;
    } catch (error) {
        console.error(`❌ Error setting up serial: ${error.message}`);
        return false;
    }
}

// ==================== DATA PARSING ====================
function parseSensorData(line) {
    if (!line || line.length === 0) return;
    
    // Store raw data for debugging
    sensorData.raw_data.push(line);
    if (sensorData.raw_data.length > 100) {
        sensorData.raw_data.shift();
    }
    
    let updated = false;
    
    // Parse pH
    const phMatch = line.match(/pH:?\s*([\d.]+)/i);
    if (phMatch) {
        const ph = parseFloat(phMatch[1]);
        sensorData.ph = ph;
        sensorData.sensor_connected = true;
        updated = true;
        
        if (ph < 6.5) {
            sensorData.ph_status = 'ACIDIC';
        } else if (ph > 7.5) {
            sensorData.ph_status = 'ALKALINE';
        } else {
            sensorData.ph_status = 'NEUTRAL';
        }
    }
    
    // Parse TDS
    const tdsMatch = line.match(/TDS\s*(?:Reading|Value)?:?\s*([\d.]+)/i);
    if (tdsMatch) {
        sensorData.tds = parseInt(parseFloat(tdsMatch[1]));
        sensorData.sensor_connected = true;
        updated = true;
    }
    
    // Parse Light
    const lightMatch = line.match(/Light:?\s*([\d.]+)/i);
    if (lightMatch) {
        sensorData.light = parseFloat(lightMatch[1]);
        sensorData.sensor_connected = true;
        updated = true;
    }
    
    // Parse MQ135
    const mq135Match = line.match(/MQ135\s*(?:Value|Reading)?:?\s*([\d.]+)/i);
    if (mq135Match) {
        sensorData.mq135 = parseInt(parseFloat(mq135Match[1]));
        sensorData.sensor_connected = true;
        updated = true;
        console.log(`📡 MQ135: ${sensorData.mq135}`);
    }
    
    // Parse Temperature
    const tempMatch = line.match(/Temp(?:erature)?:?\s*([\d.]+)/i);
    if (tempMatch) {
        sensorData.temperature = parseFloat(tempMatch[1]);
        updated = true;
    }
    
    // Parse Water Level
    if (line.includes('Water Level')) {
        if (line.includes('LOW')) {
            sensorData.water_level = 'LOW';
            sensorData.float_state = 'LOW';
            updated = true;
        } else if (line.includes('OK') || line.includes('PUMP OFF')) {
            sensorData.water_level = 'OK';
            sensorData.float_state = 'OK';
            updated = true;
        }
    }
    
    // Parse Relay States
    if (line.includes('Relay') || line.includes('PUMP')) {
        if (line.includes('FLOAT') || line.includes('Pump')) {
            sensorData.relay_states.float_relay = line.includes('ON') ? 'ON' : 'OFF';
            updated = true;
        }
        if (line.includes('MQ135') || line.includes('GAS')) {
            sensorData.relay_states.mq135_relay = line.includes('ON') ? 'ON' : 'OFF';
            updated = true;
        }
        if (line.includes('TDS') || line.includes('Dirty')) {
            sensorData.relay_states.tds_relay = line.includes('ON') ? 'ON' : 'OFF';
            updated = true;
        }
        if (line.includes('pH')) {
            sensorData.relay_states.ph_relay = line.includes('ON') ? 'ON' : 'OFF';
            updated = true;
        }
    }
    
    if (updated) {
        sensorData.last_update = Date.now();
        sensorData.data_count++;
        
        // Print every 5th update
        if (sensorData.data_count % 5 === 0) {
            console.log(`📊 Data #${sensorData.data_count}: pH=${sensorData.ph.toFixed(2)}, TDS=${sensorData.tds}, MQ135=${sensorData.mq135}`);
        }
    }
}

// ==================== API ROUTES ====================

// Get all sensor data
app.get('/api/sensors/all', (req, res) => {
    res.json({
        success: true,
        data: {
            ph: sensorData.ph,
            tds: sensorData.tds,
            light: sensorData.light,
            mq135: sensorData.mq135,
            temperature: sensorData.temperature,
            ph_status: sensorData.ph_status,
            water_level: sensorData.water_level,
            float_state: sensorData.float_state,
            relay_states: sensorData.relay_states,
            sensor_connected: sensorData.sensor_connected,
            data_count: sensorData.data_count,
            last_update: sensorData.last_update
        },
        system: {
            serial_connected: isConnected,
            port: CONFIG.comPort,
            baud_rate: CONFIG.baudRate
        }
    });
});

// Get specific sensor values
app.get('/api/sensors/ph', (req, res) => {
    res.json({
        ph: sensorData.ph,
        status: sensorData.ph_status,
        timestamp: sensorData.last_update
    });
});

app.get('/api/sensors/tds', (req, res) => {
    res.json({
        tds: sensorData.tds,
        timestamp: sensorData.last_update
    });
});

app.get('/api/sensors/mq135', (req, res) => {
    res.json({
        mq135: sensorData.mq135,
        timestamp: sensorData.last_update
    });
});

app.get('/api/sensors/light', (req, res) => {
    res.json({
        light: sensorData.light,
        timestamp: sensorData.last_update
    });
});

app.get('/api/sensors/water', (req, res) => {
    res.json({
        water_level: sensorData.water_level,
        float_state: sensorData.float_state,
        timestamp: sensorData.last_update
    });
});

app.get('/api/sensors/relays', (req, res) => {
    res.json({
        relays: sensorData.relay_states,
        timestamp: sensorData.last_update
    });
});

app.get('/api/sensors/status', (req, res) => {
    res.json({
        serial_connected: isConnected,
        sensor_connected: sensorData.sensor_connected,
        data_count: sensorData.data_count,
        last_update: sensorData.last_update,
        port: CONFIG.comPort,
        baud_rate: CONFIG.baudRate
    });
});

// Get raw serial data (for debugging)
app.get('/api/sensors/raw', (req, res) => {
    const count = parseInt(req.query.count) || 20;
    res.json({
        raw_data: sensorData.raw_data.slice(-count)
    });
});

// Reset connection
app.get('/api/sensors/reset', (req, res) => {
    if (serialPort) {
        try {
            serialPort.close();
            setTimeout(() => {
                serialPort.open();
            }, 1000);
            res.json({ success: true, message: 'Reconnecting to serial port...' });
        } catch (error) {
            res.json({ success: false, error: error.message });
        }
    } else {
        res.json({ success: false, error: 'Serial port not initialized' });
    }
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        service: 'ESP32 Sensor Server (Windows - Node.js)',
        version: '2.0',
        endpoints: {
            '/api/sensors/all': 'GET - All sensor data',
            '/api/sensors/ph': 'GET - pH only',
            '/api/sensors/tds': 'GET - TDS only',
            '/api/sensors/mq135': 'GET - MQ135 only',
            '/api/sensors/light': 'GET - Light only',
            '/api/sensors/water': 'GET - Water level',
            '/api/sensors/relays': 'GET - Relay states',
            '/api/sensors/status': 'GET - System status',
            '/api/sensors/raw': 'GET - Raw serial data (debug)',
            '/api/sensors/reset': 'GET - Reset connection'
        }
    });
});

// ==================== START SERVER ====================
console.log('='.repeat(60));
console.log('📊 ESP32 SENSOR SERVER (Windows - Node.js)');
console.log('='.repeat(60));

// Setup serial port
const serialOk = setupSerialPort();

// Get local IP address
const os = require('os');
const networkInterfaces = os.networkInterfaces();
let localIp = 'localhost';
for (const name of Object.keys(networkInterfaces)) {
    for (const net of networkInterfaces[name]) {
        if (net.family === 'IPv4' && !net.internal) {
            localIp = net.address;
            break;
        }
    }
    if (localIp !== 'localhost') break;
}

console.log('\n' + '='.repeat(60));
console.log(`🚀 Starting Sensor Server on port ${CONFIG.port}`);
console.log('='.repeat(60));
console.log(`📡 Local: http://localhost:${CONFIG.port}`);
console.log(`📡 Network: http://${localIp}:${CONFIG.port}`);
console.log('='.repeat(60));
console.log('⚠️  Make sure:');
console.log('   1. ESP32 is connected via USB');
console.log('   2. ESP32 is running your Arduino code');
console.log('   3. Close Arduino IDE Serial Monitor');
console.log('   4. COM3 is not in use by other apps');
console.log('='.repeat(60));

app.listen(CONFIG.port, '0.0.0.0', () => {
    console.log(`✅ Server running on port ${CONFIG.port}`);
    console.log(`   http://localhost:${CONFIG.port}`);
    console.log(`   http://${localIp}:${CONFIG.port}`);
});