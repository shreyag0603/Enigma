document.addEventListener('DOMContentLoaded', () => {
    const countries = [
        'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antigua and Barbuda',
        'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas, The',
        'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
        'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil',
        'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde',
        'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile',
        'China', 'Colombia', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.',
        'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cyprus', 'Czech Republic',
        'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
        'Egypt, Arab Rep.', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
        'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia, The',
        'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea',
        'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong, China', 'Hungary',
        'Iceland', 'India', 'Indonesia', 'Iran, Islamic Rep.', 'Iraq', 'Ireland',
        'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
        'Kiribati', 'Korea, Rep.', 'Kosovo', 'Kuwait', 'Kyrgyz Republic', 'Lao PDR',
        'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein',
        'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
        'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',
        'Micronesia, Fed. Sts.', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco',
        'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand',
        'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman',
        'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru',
        'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania',
        'Russian Federation', 'Rwanda', 'Samoa', 'San Marino', 'São Tomé and Principe',
        'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
        'Slovak Republic', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa',
        'South Sudan', 'Spain', 'Sri Lanka', 'St. Kitts and Nevis', 'St. Lucia',
        'St. Vincent and the Grenadines', 'Sudan', 'Suriname', 'Sweden', 'Switzerland',
        'Syrian Arab Republic', 'Taiwan, China', 'Tajikistan', 'Tanzania', 'Thailand',
        'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
        'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States',
        'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela, RB', 'Vietnam', 'West Bank and Gaza',
        'Yemen, Rep.', 'Zambia', 'Zimbabwe'
      ];
      

  const deviceTypes = {
      "Air Conditioner": ['Haier', 'Daikin', 'Samsung', 'Fujitsu', 'Mitsubishi Electric', 'Gree', 'LG Electronics', 'Panasonic', 'Carrier', 'Trane'],
      "Camera": ['Leica', 'Samsung', 'Fujifilm', 'Olympus', 'GoPro', 'Sony', 'Nikon', 'Canon', 'Panasonic', 'Pentax'],
      "Fridge": ['Haier', 'Whirlpool', 'Voltas Beko', 'Samsung', 'Hitachi', 'Bosch', 'Siemens', 'LG Electronics', 'Panasonic', 'Godrej Appliances'],
      "Vacuum Cleaner": ['Electrolux', 'Dyson', 'Kenmore', 'Shark', 'Black+Decker', 'Eureka', 'Hoover', 'Bissell', 'Miele', 'iRobot'],
      "Microwave": ['IFB', 'Whirlpool', 'Samsung', 'GE Appliances', 'Sharp', 'Haier', 'LG Electronics', 'Bosch', 'Panasonic', 'Morphy Richards'],
      "Dishwasher": ['KitchenAid', 'Bosch', 'GE Appliances', 'Miele', 'Samsung', 'Electrolux', 'Maytag', 'Whirlpool', 'LG Electronics', 'Fisher & Paykel'],
      "Security System": ['Matrix Comsec', 'Secureye', 'Top Notch Infotronix', 'Godrej Security Solutions', 'Hikvision India', 'CP Plus', 'Dahua Technology India', 'eSSL Security', 'Zicom', 'Larsen & Toubro (L&T) Technology Services'],
      "Induction": ['Bosch', 'Kenmore', 'KitchenAid', 'Samsung', 'Whirlpool', 'Electrolux', 'GE Appliances', 'Fisher & Paykel', 'Frigidaire', 'Miele'],
      "Fan": ['Crompton Greaves', 'Anchor Electricals', 'Bajaj Electricals', 'Usha', 'Havells', 'Superfan', 'Khaitan Electricals', 'Gorilla', 'Orient Electric', 'Atomberg Technologies'],
      "Smart Speaker": ['Bose', 'Amazon Echo', 'Sony', 'Harman Kardon', 'Sonos', 'JBL', 'Apple HomePod', 'Ultimate Ears', 'Google Nest', 'Bowers & Wilkins'],
      "Lights": ['Havells', 'Crompton', 'Bajaj Electricals', 'Wipro Lighting', 'Eveready', 'Surya Roshni', 'Osram', 'Philips', 'GE Lighting', 'Syska'],
      "Washing Machine": ['Voltas Beko', 'Bosch', 'Panasonic', 'Whirlpool', 'Siemens', 'IFB', 'Haier', 'Godrej Appliances', 'Samsung', 'LG Electronics'],
      "Thermostat": ['Johnson Controls', 'Schneider Electric', 'Honeywell', 'Lux Products Corporation', 'Sensi', 'Ecobee', 'Tado', 'Nest Labs', 'Siemens', 'Emerson']
  };

  const countrySelect = document.getElementById('CountryName');
  countries.forEach(country => {
      const option = document.createElement('option');
      option.value = country;
      option.textContent = country;
      countrySelect.appendChild(option);
  });

  const deviceTypeSelect = document.getElementById('DeviceType');
  Object.keys(deviceTypes).forEach(type => {
      const option = document.createElement('option');
      option.value = type;
      option.textContent = type;
      deviceTypeSelect.appendChild(option);
  });

  deviceTypeSelect.addEventListener('change', function () {
      const selectedType = this.value;
      const brandNameSelect = document.getElementById('Brand');
      brandNameSelect.disabled = false;
      brandNameSelect.innerHTML = '';

      deviceTypes[selectedType].forEach(brand => {
          const option = document.createElement('option');
          option.value = brand;
          option.textContent = brand;
          brandNameSelect.appendChild(option);
      });
  });


    // Retrieve form values
    const country = document.getElementById('CountryName').value;
    const deviceType = document.getElementById('DeviceType').value;
    const brandName = document.getElementById('Brand').value;
    const energyUsage = document.getElementById('EnergyConsumption').value;
    const deviceAge = document.getElementById('DeviceAgeMonths').value;
    const usageHours = document.getElementById('UsageHoursPerDay').value;
    const malfunctionIncidents = document.getElementById('MalfunctionIncidents').value;

    // Create a FormData object
    const formData = new FormData();
    formData.append('CountryName', country);
    formData.append('DeviceType', deviceType);
    formData.append('Brand', brandName);
    formData.append('EnergyConsumption', energyUsage);
    formData.append('DeviceAgeMonths', deviceAge);
    formData.append('UsageHoursPerDay', usageHours);
    formData.append('MalfunctionIncidents', malfunctionIncidents);

  });

  var swiper = new Swiper(".review-slider", {
    spaceBetween: 20,
    loop: true,
    autoplay: {
      delay: 3000,
      disableOnInteraction: false,
    },
    breakpoints: {
      640: {
        slidesPerView: 1,
      },
      768: {
        slidesPerView: 2,
      },
      1024: {
        slidesPerView: 4,
      },
    },
    pagination: {
      el: '.swiper-pagination',
      clickable: true,
    },
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },
  });
  