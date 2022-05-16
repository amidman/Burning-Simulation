#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <SFML/Graphics.hpp>

#define FIELDWIDTH 500
#define FIELDHEIGHT 500
#define THREDS_PER_BLOCK 10

struct Config{
    float MolarMass;    // молярна масса
    float a;            // коэффициент а в уравнении Ван Дер Ваальса
    float b;            // коэффициент b в уравнении Ван Дер Ваальса
    float dx;           // размер сетки
    float Cv;           // Молярная теплоёмкость
    float mu;           // коэффициент сжимаемрорсти
    float DiffCoef;     // коэффициент диффузии
    float ro;           // начальная плотность
    float TempKoef;     // коэффициент теплопередачи
    float R;            // газовая постоянная
    float P;            // начальное давление
    float T;            // начальная температура
	float nu;           // коэффициент вязкости
	float S;            // количество добавляемого красителя
};

Config config {18.0, 5.65, 0.031, 0.1f, 25, 0.1, 0.1f, 1, 0.1, 8.31, 1.0f, 5.0f, 50, 2};

#include "core.cu"

int main(){
	init_cuda();
	srand(time(NULL));
	sf::RenderWindow window(sf::VideoMode(FIELDWIDTH, FIELDHEIGHT), "");
	window.setFramerateLimit(60);
	window.setMouseCursorVisible(false);

	auto start = std::chrono::system_clock::now();
	auto end = std::chrono::system_clock::now();

	sf::Texture texture;
	sf::Sprite sprite;
	std::vector<sf::Uint8> pixelBuffer(FIELDWIDTH * FIELDHEIGHT * 4);
	texture.create(FIELDWIDTH, FIELDHEIGHT);

	sf::Vector2i mpos1 = { -1, -1 }, mpos2 = { -1, -1 };
	mpos2 = sf::Mouse::getPosition(window);
	bool isPressed = false;
	bool isPaused = false;
	while (window.isOpen()){
		end = std::chrono::system_clock::now();
		std::chrono::duration<float> diff = end - start;
		window.setTitle("Fluid simulator " + std::to_string(int(1.0f / diff.count())) + " fps");
		start = end;

		window.clear(sf::Color::White);
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();

			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
			{
				window.close();
			}
		}

		Vec2 Force;
		Vec2 ForcePoint;

		mpos1 = sf::Mouse::getPosition(window);

		//Force.x  = (mpos1.x - mpos2.x) * 1;
		//Force.y  = (mpos1.y - mpos2.y) * 1;

		Force.x  = 10;
		Force.y  = 0;

		ForcePoint.x  = mpos1.x;
		ForcePoint.y  = mpos1.y;
		
		//ForcePoint.x  = 1;
		//ForcePoint.y  = 1;

		std::swap(mpos1, mpos2);

		float dt = 0.0001;

		RenderImage(pixelBuffer.data(), dt, Force, ForcePoint, 1);

		texture.update(pixelBuffer.data());
		sprite.setTexture(texture);
		sprite.setScale({ 1, 1 });
		window.draw(sprite);
		window.display();
	}
	Exit_cuda();
	return 0;
}