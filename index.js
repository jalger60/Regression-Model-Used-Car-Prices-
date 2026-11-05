let pageData;

function fillBox(id, values) {
    let box = document.getElementById(id);
    box.replaceChildren();

    for (let value of values) {
        box.add(new Option(value === "–" ? "Not listed" : value, value));
    }
}

function updateModels() {
    let brand = document.getElementById('box1').value;
    fillBox('box2', Object.keys(pageData.catalog[brand]));
    updateCarChoices();
}

function updateCarChoices() {
    let brand = document.getElementById('box1').value;
    let model = document.getElementById('box2').value;
    let choices = pageData.catalog[brand][model];

    fillBox('box5', choices.fuel_type);
    fillBox('box6', choices.engine);
    fillBox('box7', choices.transmission);
}

// Builds the same 265 inputs as the notebook's preprocessor
function encodeCar(car) {
    let x = new Float32Array(pageData.input_dim);

    car.age = pageData.age_reference_year - car.model_year;

    pageData.num_cols.forEach((column, index) => {
        x[index] = car[column];
    });

    for (let [column, rule] of Object.entries(pageData.encoding)) {
        let position = rule.categories.indexOf(car[column]);

        if (position >= 0) {
            x[rule.start + position] = 1;
        } else if (rule.infrequent) {
            x[rule.start + rule.categories.length] = 1;
        }
    }

    return x;
}

function showError(message) {
    let error = document.createElement('p');
    error.className = 'error';
    error.textContent = message;
    document.getElementById('predictions').replaceChildren(error);
}

async function runExample() {

    let [minYear, maxYear] = pageData.year_range;
    let year = document.getElementById('box3').value;
    let mileage = document.getElementById('box4').value;

    if (year === '' || year < minYear || year > maxYear) {
        showError(`Enter a model year from ${minYear} to ${maxYear}.`);
        return;
    }
    if (mileage === '' || mileage < 0 || mileage > pageData.max_mileage) {
        showError(`Enter a mileage from 0 to ${pageData.max_mileage}.`);
        return;
    }

    let car = {
        brand:        document.getElementById('box1').value,
        model:        document.getElementById('box2').value,
        model_year:   Number(year),
        milage:       Number(mileage),
        fuel_type:    document.getElementById('box5').value,
        engine:       document.getElementById('box6').value,
        transmission: document.getElementById('box7').value,
        ext_col:      document.getElementById('box8').value,
        int_col:      document.getElementById('box9').value,
        accident:     document.getElementById('box10').value
    };

    try {
        let x = encodeCar(car);

        let tensorX = new onnx.Tensor(x, 'float32', [1, x.length]);

        let session = new onnx.InferenceSession();

        await session.loadModel("./used_car_prices.onnx");
        let outputMap = await session.run([tensorX]);
        let outputData = outputMap.get('output1');

        // The model outputs log(1 + price)
        let predictedPrice = Math.expm1(outputData.data[0]);

        let predictions = document.getElementById('predictions');

        predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
        <table>
          <tr>
            <td>  Predicted Car Price  </td>
            <td id="td0" class="price">  ${predictedPrice.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })}  </td>
          </tr>
        </table>`;
    } catch (error) {
        console.error(error);
        showError(`The prediction failed. ${error.message}`);
    }
}

async function setupPage() {
    try {
        let response = await fetch("./used_car_page_data.json");
        if (!response.ok) {
            throw new Error(`Could not load used_car_page_data.json (HTTP ${response.status}).`);
        }
        pageData = await response.json();

        fillBox('box1', Object.keys(pageData.catalog));
        fillBox('box8', pageData.options.ext_col);
        fillBox('box9', pageData.options.int_col);
        fillBox('box10', pageData.options.accident);

        document.getElementById('box1').value = "Toyota";
        updateModels();

        document.getElementById('box1').addEventListener('change', updateModels);
        document.getElementById('box2').addEventListener('change', updateCarChoices);
        document.getElementById('predictButton').disabled = false;
    } catch (error) {
        console.error(error);
        showError(`The page could not load. ${error.message}`);
    }
}

setupPage();
