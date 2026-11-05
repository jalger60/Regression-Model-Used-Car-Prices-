const NEURAL_NET_MODEL = "./used_car_prices.onnx";
const NEURAL_NET_WEIGHTS = "./used_car_prices.onnx.data";
const XGBOOST_MODEL = "./xgboost_used_car_prices.onnx";
const PAGE_DATA = "./used_car_page_data.json";

let pageData;
let neuralNetSession;
let xgboostSession;

// Set by "Try a Test Car", and cleared when the form changes
let activeTestCar = null;

const fields = {
    brand: document.getElementById("f_brand"),
    model: document.getElementById("f_model"),
    model_year: document.getElementById("f_year"),
    milage: document.getElementById("f_mileage"),
    fuel_type: document.getElementById("f_fuel"),
    engine: document.getElementById("f_engine"),
    transmission: document.getElementById("f_transmission"),
    ext_col: document.getElementById("f_ext_color"),
    int_col: document.getElementById("f_int_color"),
    accident: document.getElementById("f_accident")
};

function setStatus(message, isError = false) {
    const status = document.getElementById("status");
    status.textContent = message;
    status.classList.toggle("error", isError);
}

function setOptions(select, values) {
    select.replaceChildren(...values.map((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value === "–" ? "Not listed" : value;
        return option;
    }));
}

function updateModelOptions() {
    setOptions(fields.model, Object.keys(pageData.catalog[fields.brand.value]));
    updateCarOptions();
}

function updateCarOptions() {
    const choices = pageData.catalog[fields.brand.value][fields.model.value];
    setOptions(fields.fuel_type, choices.fuel_type);
    setOptions(fields.engine, choices.engine);
    setOptions(fields.transmission, choices.transmission);
}

function clearResult() {
    activeTestCar = null;
    document.getElementById("predictions").replaceChildren();
    setStatus("");
}

function setupForm() {
    setOptions(fields.brand, Object.keys(pageData.catalog));
    setOptions(fields.ext_col, pageData.options.ext_col);
    setOptions(fields.int_col, pageData.options.int_col);
    setOptions(fields.accident, pageData.options.accident);

    if ("Toyota" in pageData.catalog) {
        fields.brand.value = "Toyota";
    }
    updateModelOptions();

    fields.brand.addEventListener("change", updateModelOptions);
    fields.model.addEventListener("change", updateCarOptions);

    Object.values(fields).forEach((field) => {
        field.addEventListener("change", clearResult);
    });

    const [minYear, maxYear] = pageData.year_range;
    fields.model_year.min = minYear;
    fields.model_year.max = maxYear;
    fields.milage.max = pageData.max_mileage;
}

function readCar() {
    return {
        brand: fields.brand.value,
        model: fields.model.value,
        model_year: Number(fields.model_year.value),
        milage: Number(fields.milage.value),
        fuel_type: fields.fuel_type.value,
        engine: fields.engine.value,
        transmission: fields.transmission.value,
        ext_col: fields.ext_col.value,
        int_col: fields.int_col.value,
        accident: fields.accident.value
    };
}

function validateCar(car) {
    const [minYear, maxYear] = pageData.year_range;

    if (fields.model_year.value === "" || !Number.isInteger(car.model_year)
        || car.model_year < minYear || car.model_year > maxYear) {
        return `Enter a model year from ${minYear} to ${maxYear}.`;
    }
    if (fields.milage.value === "" || !Number.isFinite(car.milage)
        || car.milage < 0 || car.milage > pageData.max_mileage) {
        return `Enter a mileage from 0 to ${pageData.max_mileage.toLocaleString("en-US")}.`;
    }
    return "";
}

// Builds the same inputs as the notebook's `preprocessor`
function encodeCar(car) {
    const x = new Float32Array(pageData.input_dim);
    const numbers = {
        model_year: car.model_year,
        milage: car.milage,
        age: pageData.age_reference_year - car.model_year
    };

    pageData.num_cols.forEach((column, index) => {
        x[index] = numbers[column];
    });

    Object.entries(pageData.encoding).forEach(([column, rule]) => {
        const position = rule.categories.indexOf(car[column]);

        if (position >= 0) {
            x[rule.start + position] = 1;
        } else if (rule.infrequent) {
            // The infrequent column comes right after the listed categories
            x[rule.start + rule.categories.length] = 1;
        }
    });

    return x;
}

async function predictPrice(session, x) {
    const tensorX = new ort.Tensor("float32", x, [1, x.length]);
    const outputMap = await session.run({ [session.inputNames[0]]: tensorX });
    const logPrice = outputMap[session.outputNames[0]].data[0];

    // The models output log(1 + price)
    return Math.expm1(logPrice);
}

function formatPrice(price) {
    return price.toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0
    });
}

function addResultRow(table, label, value, isPrice) {
    const row = table.insertRow();
    row.insertCell().textContent = label;

    const valueCell = row.insertCell();
    valueCell.textContent = value;
    valueCell.classList.toggle("price", isPrice);
}

function showResults(neuralNetPrice, xgboostPrice) {
    const table = document.createElement("table");
    const header = table.createTHead().insertRow();

    ["Model", "Predicted Price"].forEach((title) => {
        const cell = document.createElement("th");
        cell.textContent = title;
        header.appendChild(cell);
    });

    addResultRow(table, "Neural Network", formatPrice(neuralNetPrice), true);
    addResultRow(table, "XGBoost", formatPrice(xgboostPrice), true);

    if (activeTestCar) {
        addResultRow(table, "Actual Listed Price", formatPrice(activeTestCar.price), false);
    }

    document.getElementById("predictions").replaceChildren(table);
}

async function runExample() {
    const car = readCar();
    const problem = validateCar(car);

    if (problem) {
        document.getElementById("predictions").replaceChildren();
        setStatus(problem, true);
        return;
    }

    try {
        const x = encodeCar(car);
        const neuralNetPrice = await predictPrice(neuralNetSession, x);
        const xgboostPrice = await predictPrice(xgboostSession, x);

        setStatus("");
        showResults(neuralNetPrice, xgboostPrice);
    } catch (error) {
        console.error(error);
        setStatus("The prediction failed. Reload the page and try again.", true);
    }
}

async function tryTestCar() {
    const cars = pageData.test_cars;
    const car = cars[Math.floor(Math.random() * cars.length)];

    fields.brand.value = car.brand;
    updateModelOptions();
    fields.model.value = car.model;
    updateCarOptions();

    fields.model_year.value = car.model_year;
    fields.milage.value = car.milage;
    fields.fuel_type.value = car.fuel_type;
    fields.engine.value = car.engine;
    fields.transmission.value = car.transmission;
    fields.ext_col.value = car.ext_col;
    fields.int_col.value = car.int_col;
    fields.accident.value = car.accident;

    activeTestCar = car;
    await runExample();
}

// Loads the model file, plus its weights file if the export created one
async function loadNeuralNet() {
    const options = { executionProviders: ["wasm"] };
    const weights = await fetch(NEURAL_NET_WEIGHTS, { method: "HEAD" });

    if (weights.ok) {
        options.externalData = [{
            path: NEURAL_NET_WEIGHTS.replace("./", ""),
            data: NEURAL_NET_WEIGHTS
        }];
    }

    return ort.InferenceSession.create(NEURAL_NET_MODEL, options);
}

async function loadPage() {
    try {
        if (typeof ort === "undefined") {
            throw new Error("ONNX Runtime did not load.");
        }
        ort.env.wasm.numThreads = 1;

        const response = await fetch(PAGE_DATA);
        if (!response.ok) {
            throw new Error(`Could not load ${PAGE_DATA}.`);
        }
        pageData = await response.json();
        setupForm();

        neuralNetSession = await loadNeuralNet();
        xgboostSession = await ort.InferenceSession.create(XGBOOST_MODEL, { executionProviders: ["wasm"] });

        document.getElementById("predictButton").disabled = false;
        document.getElementById("testCarButton").disabled = false;
        setStatus("");
    } catch (error) {
        console.error(error);
        setStatus(`The page could not load the models. ${error.message}`, true);
    }
}

loadPage();